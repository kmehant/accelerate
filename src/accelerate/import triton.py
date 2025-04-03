from trl import SFTConfig, SFTTrainer
args = SFTConfig(gradient_checkpointing=True)
SFTTrainer(args=args)
