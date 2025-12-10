import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# import k2
import sentencepiece as spm
import torch
import torch.nn as nn
import torchaudio
from datasets import load_from_disk, concatenate_datasets
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ICEFALL_ROOT = Path(__file__).parent / "icefall"
sys.path.insert(0, str(ICEFALL_ROOT / "egs/librispeech/ASR/zipformer"))
sys.path.insert(0, str(ICEFALL_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

from zipformer import Zipformer2
from decoder import Decoder
from joiner import Joiner
from subsampling import Conv2dSubsampling
from model import AsrModel
from scaling import ScheduledFloat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass 
class ModelConfig:
    """
    Model architecture for ~45M params.
    Larger than 26M pretrained but optimized for target size.
    """
    # Encoder
    num_encoder_layers: str = "2,2,3,3,2,2"  # 14 layers total
    downsampling_factor: str = "1,2,4,8,4,2"
    encoder_dim: str = "192,288,384,384,288,192"  # Balanced dims
    feedforward_dim: str = "512,768,1024,1024,768,512"  # Balanced FFN
    num_heads: str = "4,4,4,8,4,4"
    encoder_unmasked_dim: str = "192,192,256,256,192,192"
    query_head_dim: int = 32
    value_head_dim: int = 12
    pos_head_dim: int = 4
    pos_dim: int = 48
    cnn_module_kernel: str = "31,31,15,15,15,31"
    
    # Decoder & Joiner
    decoder_dim: int = 512
    joiner_dim: int = 512
    context_size: int = 2
    
    # Vocab (keep same BPE)
    vocab_size: int = 2000
    blank_id: int = 0
    
    # Other
    causal: bool = False
    feature_dim: int = 80


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Paths
    pretrained_jit: str = "pretrained_model/jit_script.pt"
    bpe_model: str = "pretrained_model/bpe.model"
    exp_dir: str = "exp_finetune_large"
    
    # Training params  
    num_epochs: int = 10
    batch_size: int = 4
    lr: float = 5e-5  # Lower LR for larger model
    weight_decay: float = 1e-6
    grad_clip: float = 5.0
    warmup_steps: int = 1000
    
    # RNNT loss params
    prune_range: int = 5
    am_scale: float = 0.0
    lm_scale: float = 0.0
    simple_loss_scale: float = 0.5
    
    # Audio params
    sample_rate: int = 16000
    max_duration: float = 20.0
    
    # Other
    use_fp16: bool = True
    num_workers: int = 4
    save_every_epoch: int = 1
    log_every: int = 1
    seed: int = 42


def _to_int_tuple(s: str) -> tuple:
    return tuple(map(int, s.split(",")))


def create_large_model(config: ModelConfig) -> AsrModel:
    """Create larger ZipFormer model (40-50M params)."""
    
    encoder_embed = Conv2dSubsampling(
        in_channels=config.feature_dim,
        out_channels=_to_int_tuple(config.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(config.downsampling_factor),
        num_encoder_layers=_to_int_tuple(config.num_encoder_layers),
        encoder_dim=_to_int_tuple(config.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(config.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(str(config.query_head_dim)),
        pos_head_dim=_to_int_tuple(str(config.pos_head_dim)),
        value_head_dim=_to_int_tuple(str(config.value_head_dim)),
        pos_dim=config.pos_dim,
        num_heads=_to_int_tuple(config.num_heads),
        feedforward_dim=_to_int_tuple(config.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(config.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=config.causal,
    )
    
    decoder = Decoder(
        vocab_size=config.vocab_size,
        decoder_dim=config.decoder_dim,
        blank_id=config.blank_id,
        context_size=config.context_size,
    )
    
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(config.encoder_dim)),
        decoder_dim=config.decoder_dim,
        joiner_dim=config.joiner_dim,
        vocab_size=config.vocab_size,
    )
    
    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=max(_to_int_tuple(config.encoder_dim)),
        decoder_dim=config.decoder_dim,
        vocab_size=config.vocab_size,
        use_transducer=True,
        use_ctc=False,
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained_encoder_weights(model: AsrModel, jit_path: str) -> int:
    """
    Load encoder weights from pretrained model where shapes match.
    For larger model, only compatible layers will be loaded.
    """
    logger.info(f"Loading compatible weights from {jit_path}")
    
    jit_model = torch.jit.load(jit_path, map_location='cpu')
    jit_state_dict = jit_model.state_dict()
    model_state_dict = model.state_dict()
    
    loaded_count = 0
    
    for jit_key, jit_value in jit_state_dict.items():
        # Map keys
        model_key = jit_key
        if jit_key.startswith("encoder.encoder_embed."):
            model_key = jit_key.replace("encoder.encoder_embed.", "encoder_embed.")
        elif jit_key.startswith("encoder.encoder."):
            model_key = jit_key.replace("encoder.encoder.", "encoder.")
        
        if model_key in model_state_dict:
            if model_state_dict[model_key].shape == jit_value.shape:
                model_state_dict[model_key] = jit_value
                loaded_count += 1
    
    model.load_state_dict(model_state_dict)
    logger.info(f"Loaded {loaded_count} compatible weights from pretrained model")
    
    return loaded_count


class MultiDataset(Dataset):
    """Dataset combining multiple Vietnamese ASR datasets."""
    
    def __init__(
        self,
        data_configs: List[dict],
        sp: spm.SentencePieceProcessor,
        sample_rate: int = 16000,
        max_duration: float = 20.0,
    ):
        self.sp = sp
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_frames = int(max_duration * sample_rate)
        
        # Store datasets and their configs
        self.datasets = []
        self.text_cols = []
        self.offsets = [0]
        
        total_samples = 0
        for cfg in data_configs:
            path = cfg["path"]
            text_col = cfg["text_col"]
            name = cfg.get("name", path)
            
            # Load dataset
            if os.path.exists(os.path.join(path, "data")):
                ds = load_from_disk(os.path.join(path, "data"))
            else:
                ds = load_from_disk(path)
            
            self.datasets.append(ds)
            self.text_cols.append(text_col)
            total_samples += len(ds)
            self.offsets.append(total_samples)
            
            logger.info(f"Loaded {len(ds)} samples from {name} (text_col={text_col})")
        
        self.total_samples = total_samples
        logger.info(f"Total training samples: {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def _get_sample(self, idx):
        """Get sample from correct dataset based on index."""
        for i, offset in enumerate(self.offsets[:-1]):
            if idx < self.offsets[i + 1]:
                local_idx = idx - offset
                return self.datasets[i][local_idx], self.text_cols[i]
        raise IndexError(f"Index {idx} out of range")
    
    def __getitem__(self, idx):
        # sample, text_col = self._get_sample(idx)
        # audio = sample["audio"]
        # text = sample[text_col]
        
        # # Get waveform
        # if isinstance(audio, dict):
        #     waveform = torch.tensor(audio["array"], dtype=torch.float32)
        #     sr = audio["sampling_rate"]
        # else:
        #     waveform = torch.tensor(audio, dtype=torch.float32)
        #     sr = self.sample_rate
        sample, text_col = self._get_sample(idx)
        audio = sample["audio"]
        text = sample[text_col]

        # Decode waveform + sampling rate
        if hasattr(audio, "get_all_samples"):  # TorchCodec AudioDecoder
            samples = audio.get_all_samples()
            waveform = samples.data  # torch.Tensor [channels, time]
            sr = samples.sample_rate
        elif isinstance(audio, dict) and "array" in audio:  # HF old format
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            sr = audio.get("sampling_rate", self.sample_rate)
        else:
            # Last fallback (only if it's already a tensor/array)
            waveform = torch.tensor(audio, dtype=torch.float32)
            sr = getattr(self, "sample_rate", None)

        # Squeeze channel dim if mono [1, T] → [T]
        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Truncate if too long
        if waveform.size(0) > self.max_frames:
            waveform = waveform[:self.max_frames]
        
        # Extract features
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        features = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=self.sample_rate,
            frame_length=25.0,
            frame_shift=10.0,
        )
        
        # Encode text (UPPERCASE for this tokenizer)
        text_upper = text.upper().strip()
        tokens = self.sp.encode(text_upper, out_type=int)
        
        return {
            "features": features,
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    features = [x["features"] for x in batch]
    tokens = [x["tokens"] for x in batch]  # Keep as list of tensors
    
    # Pad features
    max_feat_len = max(f.size(0) for f in features)
    padded_features = torch.zeros(len(features), max_feat_len, 80)
    feature_lens = torch.zeros(len(features), dtype=torch.long)
    
    for i, f in enumerate(features):
        padded_features[i, :f.size(0)] = f
        feature_lens[i] = f.size(0)
    
    return {
        "features": padded_features,
        "feature_lens": feature_lens,
        "tokens": tokens,  # List of tensors for k2.RaggedTensor
    }


# def compute_loss(
#     model: AsrModel,
#     batch: dict,
#     device: torch.device,
#     prune_range: int = 5,
#     am_scale: float = 0.0,
#     lm_scale: float = 0.0,
#     simple_loss_scale: float = 0.5,
# ) -> Tuple[torch.Tensor, dict]:
#     """Compute RNNT loss using k2."""
    
#     features = batch["features"].to(device)
#     feature_lens = batch["feature_lens"].to(device)
#     tokens = batch["tokens"]  # List of tensors
    
#     # Create k2 RaggedTensor from list of tensors
#     y_list = [t.tolist() for t in tokens]
#     y = k2.RaggedTensor(y_list).to(device)
    
#     # Forward pass using model's forward method
#     simple_loss, pruned_loss, _, _, _ = model(
#         x=features,
#         x_lens=feature_lens,
#         y=y,
#         prune_range=prune_range,
#         am_scale=am_scale,
#         lm_scale=lm_scale,
#     )
    
#     # Total loss
#     total_loss = simple_loss_scale * simple_loss + pruned_loss
    
#     loss_info = {
#         "simple_loss": simple_loss.item(),
#         "pruned_loss": pruned_loss.item(),
#         "total_loss": total_loss.item(),
#     }
    
#     return total_loss, loss_info

from typing import Tuple, Dict, List
import torch
from torch import Tensor


import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict

def compute_loss(
    model: AsrModel,
    batch: dict,
    device: torch.device,
    prune_range: int = 5,
    am_scale: float = 0.0,
    lm_scale: float = 0.0,
    simple_loss_scale: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Compute RNNT loss without k2."""

    # Move features to device
    features = batch["features"].to(device)
    feature_lens = batch["feature_lens"].to(device)

    # `tokens` is already a list of 1-D tensors
    # Pass it through as `y` exactly as in original API
    y = batch["tokens"]  # NO k2 conversion

    # Forward pass — keep all original arguments
    simple_loss, pruned_loss, _, _, _ = model(
        x=features,
        x_lens=feature_lens,
        y=y,
        prune_range=prune_range,
        am_scale=am_scale,
        lm_scale=lm_scale,
    )

    # Combine losses
    total_loss = simple_loss_scale * simple_loss + pruned_loss

    # Log info
    loss_info = {
        "simple_loss": float(simple_loss.detach()),
        "pruned_loss": float(pruned_loss.detach()),
        "total_loss": float(total_loss.detach()),
    }

    return total_loss, loss_info
def train_epoch(
    model: AsrModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        with autocast(enabled=config.use_fp16):
            loss, loss_info = compute_loss(
                model, batch, device,
                prune_range=config.prune_range,
                am_scale=config.am_scale,
                lm_scale=config.lm_scale,
                simple_loss_scale=config.simple_loss_scale,
            )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss_info["total_loss"]
        num_batches += 1
        
        if batch_idx % config.log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss_info['total_loss']:.4f}",
                "simple": f"{loss_info['simple_loss']:.4f}",
                "pruned": f"{loss_info['pruned_loss']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: AsrModel,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        loss, _ = compute_loss(
            model, batch, device,
            prune_range=config.prune_range,
        )
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    # Configs
    model_config = ModelConfig()
    train_config = TrainingConfig()
    train_config.num_epochs = args.num_epochs
    train_config.batch_size = args.batch_size
    train_config.lr = args.lr
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model (40-50M params)...")
    model = create_large_model(model_config)
    
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Load pretrained weights (only compatible ones)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Resumed from {args.resume}")
    else:
        # Try to load compatible weights from pretrained
        load_pretrained_encoder_weights(model, train_config.pretrained_jit)
    
    model = model.to(device)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(train_config.bpe_model)
    logger.info(f"Loaded BPE model with vocab size {sp.get_piece_size()}")
    
    # Dataset configs - prioritize dialect and loanword data
    train_data_configs = [
        # ViMD: Best for dialects + English loanwords (HIGHEST PRIORITY)
        # {
        #     "path": "data/vimd/train",
        #     "text_col": "text",
        #     "name": "vimd",
        #     "weight": 2.0,  # Higher weight
        # },
        # # VLSP2020: Large diverse dataset
        # {
        #     "path": "data/vlsp2020/train",
        #     "text_col": "transcription",
        #     "name": "vlsp2020",
        #     "weight": 1.0,
        # },
        # # InfoRe1: Clean audio
        {
            "path": "/Users/dungnguyen/Downloads/finetune_zipformer/data/infore1/train/data/train/",
            "text_col": "transcription",
            "name": "infore1",
            "weight": 1.0,
        },
        # VIVOS: Standard Vietnamese
        # {
        #     "path": "data/vivos/train",
        #     "text_col": "sentence",
        #     "name": "vivos",
        #     "weight": 1.0,
        # },
    ]
    
    val_data_configs = [
                {
            "path": "/Users/dungnguyen/Downloads/finetune_zipformer/data/infore1/train/data/train/",
            "text_col": "transcription",
            "name": "infore1",
            "weight": 1.0,
        },

    ]
    
    # Create datasets
    logger.info("Loading training data...")
    train_dataset = MultiDataset(
        train_data_configs,
        sp,
        sample_rate=train_config.sample_rate,
        max_duration=train_config.max_duration,
    )
    
    logger.info("Loading validation data...")
    val_dataset = MultiDataset(
        val_data_configs,
        sp,
        sample_rate=train_config.sample_rate,
        max_duration=train_config.max_duration,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    
    total_steps = len(train_loader) * train_config.num_epochs
    warmup_steps = train_config.warmup_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=train_config.use_fp16)
    
    # Create exp dir
    os.makedirs(train_config.exp_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info(f"Model: {num_params/1e6:.1f}M params")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {train_config.batch_size}")
    logger.info(f"Epochs: {train_config.num_epochs}")
    logger.info(f"Learning rate: {train_config.lr}")
    logger.info("=" * 60)
    
    for epoch in range(1, train_config.num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{train_config.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, train_config
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device, train_config)
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        
        torch.save(checkpoint, f"{train_config.exp_dir}/epoch_{epoch}.pt")
        torch.save(model.state_dict(), f"{train_config.exp_dir}/latest.pt")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{train_config.exp_dir}/best.pt")
            logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {train_config.exp_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
