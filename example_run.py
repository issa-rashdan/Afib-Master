import torch
from model import KanResWideX


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = KanResWideX(input_channels=1, output_size=4)
    model = model.to(device)

    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 8
    sequence_length = 1000
    input_channels = 1

    dummy_input = torch.randn(batch_size, input_channels, sequence_length).to(device)
    print(f"\nInput shape: {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[0]}")


if __name__ == "__main__":
    main()
