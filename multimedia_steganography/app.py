from __future__ import annotations

import tempfile
import zlib
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

# Reuse the tested functions in your existing src/steganography.py.
from src.steganography import (
    HEADER_SIZE,
    encode_image,
    decode_image,
    encrypt_payload,
    decrypt_payload,
)


def pack_file(file_path: str) -> bytes:
    """Package the original filename and file bytes, then compress them."""
    path = Path(file_path)
    filename = path.name.encode("utf-8")
    if len(filename) > 65535:
        raise ValueError("Filename is too long.")

    # Format: 2-byte filename length | UTF-8 filename | original file bytes
    package = len(filename).to_bytes(2, "big") + filename + path.read_bytes()
    return zlib.compress(package, level=9)


def unpack_file(compressed_payload: bytes) -> tuple[bytes, str]:
    """Decompress the package and recover the original filename and contents."""
    package = zlib.decompress(compressed_payload)
    if len(package) < 2:
        raise ValueError("Recovered package is invalid.")

    name_length = int.from_bytes(package[:2], "big")
    if len(package) < 2 + name_length:
        raise ValueError("Recovered filename metadata is invalid.")

    filename = package[2:2 + name_length].decode("utf-8")
    filename = Path(filename.replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        filename = "recovered_file.bin"

    return package[2 + name_length:], filename


def image_capacity_bytes(image_path: str) -> int:
    with Image.open(image_path) as image:
        width, height = image.size
    # 3 LSBs per pixel, minus the steganography header.
    return max(0, (width * height * 3) // 8 - HEADER_SIZE)


def hide_file(
    cover_path: Optional[str],
    secret_path: Optional[str],
    password: str,
):
    if not cover_path:
        raise gr.Error("Please upload a PNG cover image.")
    if not secret_path:
        raise gr.Error("Please upload a file to hide.")
    if not password:
        raise gr.Error("Please enter an encryption password.")

    try:
        with Image.open(cover_path) as image:
            if image.format != "PNG":
                raise gr.Error(
                    "The cover image must be PNG to preserve the hidden bits."
                )

        original_size = Path(secret_path).stat().st_size
        compressed = pack_file(secret_path)
        encrypted = encrypt_payload(compressed, password)
        capacity = image_capacity_bytes(cover_path)

        if len(encrypted) > capacity:
            raise gr.Error(
                "Payload is too large for this cover image.\n"
                f"Encrypted payload: {len(encrypted):,} bytes\n"
                f"Available capacity: {capacity:,} bytes\n"
                "Choose a larger PNG image or a smaller file."
            )

        out = tempfile.NamedTemporaryFile(
            prefix="stego_", suffix=".png", delete=False
        )
        out_path = out.name
        out.close()

        encode_image(cover_path, out_path, encrypted)

        compression_ratio = len(compressed) / max(1, original_size) * 100
        status = (
            "Embedding successful.\n"
            f"Original file: {Path(secret_path).name}\n"
            f"Original size: {original_size:,} bytes\n"
            f"Compressed package: {len(compressed):,} bytes\n"
            f"Encrypted payload: {len(encrypted):,} bytes\n"
            f"Available image capacity: {capacity:,} bytes\n"
            f"Compressed/original size ratio: {compression_ratio:.2f}%\n"
            "The password is not stored in the image."
        )
        return out_path, status, out_path

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Could not hide file: {exc}") from exc


def extract_file(stego_path: Optional[str], password: str):
    if not stego_path:
        raise gr.Error("Please upload the stego PNG image.")
    if not password:
        raise gr.Error("Please enter the decryption password.")

    try:
        encrypted = decode_image(stego_path)
        compressed = decrypt_payload(encrypted, password)
        file_bytes, filename = unpack_file(compressed)

        suffix = Path(filename).suffix or ".bin"
        out = tempfile.NamedTemporaryFile(
            prefix="recovered_", suffix=suffix, delete=False
        )
        out_path = out.name
        out.close()
        Path(out_path).write_bytes(file_bytes)

        status = (
            "Extraction successful.\n"
            f"Recovered filename: {filename}\n"
            f"Recovered size: {len(file_bytes):,} bytes\n"
            "AES-GCM authentication succeeded."
        )
        return out_path, status

    except Exception as exc:
        raise gr.Error(
            "Extraction failed. Check that the image contains a valid payload, "
            "the password is correct, and the image has not been modified."
        ) from exc


with gr.Blocks(title="Secure Image Steganography") as demo:
    gr.Markdown(
        "# Secure Image Steganography\n"
        "Compress a file, encrypt it with AES-GCM, and hide it inside a PNG image. "
        "The receiver can extract, decrypt, and recover the original file."
    )

    with gr.Tab("Hide a file"):
        with gr.Row():
            with gr.Column():
                cover_input = gr.Image(
                    label="Cover image (PNG only)", type="filepath"
                )
                secret_input = gr.File(
                    label="File to hide (any file type)", type="filepath"
                )
                hide_password = gr.Textbox(
                    label="Encryption password", type="password"
                )
                hide_button = gr.Button(
                    "Compress, encrypt, and hide", variant="primary"
                )
            with gr.Column():
                stego_output = gr.Image(
                    label="Generated stego image", type="filepath"
                )
                stego_download = gr.File(label="Download stego PNG")
                hide_status = gr.Textbox(label="Results", lines=8)

        hide_button.click(
            fn=hide_file,
            inputs=[cover_input, secret_input, hide_password],
            outputs=[stego_output, hide_status, stego_download],
        )

    with gr.Tab("Extract a file"):
        with gr.Row():
            with gr.Column():
                stego_input = gr.Image(
                    label="Stego image (PNG)", type="filepath"
                )
                extract_password = gr.Textbox(
                    label="Decryption password", type="password"
                )
                extract_button = gr.Button(
                    "Extract, decrypt, and recover", variant="primary"
                )
            with gr.Column():
                recovered_output = gr.File(label="Download recovered file")
                extract_status = gr.Textbox(label="Results", lines=6)

        extract_button.click(
            fn=extract_file,
            inputs=[stego_input, extract_password],
            outputs=[recovered_output, extract_status],
        )

    gr.Markdown(
        "**Important:** Use PNG throughout. JPEG compression or image-editing "
        "services can destroy hidden data. AES-GCM protects the payload, but "
        "does not conceal the fact that steganography may have been used. "
        "Keep the password safe."
    )


if __name__ == "__main__":
    demo.launch()
