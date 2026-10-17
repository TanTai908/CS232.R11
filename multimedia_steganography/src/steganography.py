import os
import getpass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.exceptions import InvalidTag
from pathlib import Path
import zlib
from PIL import Image

MAGIC = b"STEG"
HEADER_SIZE = len(MAGIC) + 4  # 4-byte payload length


def bytes_to_bits(data: bytes):
    for byte in data:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


def bits_to_bytes(bits):
    result = bytearray()

    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]

        if len(chunk) != 8:
            break

        value = 0
        for bit in chunk:
            value = (value << 1) | bit

        result.append(value)

    return bytes(result)


def encode_image(cover_path, output_path, payload: bytes):
    image = Image.open(cover_path).convert("RGB")

    # Header makes the payload identifiable and records its length.
    header = MAGIC + len(payload).to_bytes(4, "big")
    data = header + payload
    bits = list(bytes_to_bits(data))

    capacity_bits = image.width * image.height * 3

    if len(bits) > capacity_bits:
        raise ValueError(
            f"Payload too large: {len(bits)} bits required, "
            f"{capacity_bits} bits available."
        )

    pixels = list(image.get_flattened_data())
    result = []
    bit_index = 0

    for pixel in pixels:
        channels = list(pixel)

        for channel in range(3):
            if bit_index < len(bits):
                channels[channel] = (
                    (channels[channel] & 0xFE) | bits[bit_index]
                )
                bit_index += 1

        result.append(tuple(channels))

    encoded = Image.new("RGB", image.size)
    encoded.putdata(result)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    encoded.save(output_path, format="PNG")


def decode_image(image_path) -> bytes:
    image = Image.open(image_path).convert("RGB")

    # Read the header first: 8 bytes = magic + payload length.
    header_bits = []

    for pixel in image.get_flattened_data():
        for channel in pixel:
            header_bits.append(channel & 1)

            if len(header_bits) == HEADER_SIZE * 8:
                break

        if len(header_bits) == HEADER_SIZE * 8:
            break

    header = bits_to_bytes(header_bits)

    if len(header) != HEADER_SIZE or header[:4] != MAGIC:
        raise ValueError("No valid STEG payload found in this image.")

    payload_size = int.from_bytes(header[4:8], "big")
    total_bytes = HEADER_SIZE + payload_size
    capacity_bytes = image.width * image.height * 3 // 8

    if total_bytes > capacity_bytes:
        raise ValueError("Invalid payload length in image header.")

    bits = []
    required_bits = total_bytes * 8

    for pixel in image.get_flattened_data():
        for channel in pixel:
            bits.append(channel & 1)

            if len(bits) == required_bits:
                break

        if len(bits) == required_bits:
            break

    recovered = bits_to_bytes(bits)
    return recovered[HEADER_SIZE:]
def get_capacity_bytes(image_path):
    image = Image.open(image_path).convert("RGB")
    total_bits = image.width * image.height * 3
    return total_bits // 8 - HEADER_SIZE


SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit AES key from a password."""
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2**14,
        r=8,
        p=1,
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt_payload(data: bytes, password: str) -> bytes:
    """Encrypt data and package salt + nonce + ciphertext/tag."""
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)

    key = derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)

    # AESGCM.encrypt includes the 16-byte authentication tag.
    return salt + nonce + ciphertext


def decrypt_payload(payload: bytes, password: str) -> bytes:
    """Recover and authenticate an encrypted payload."""
    minimum_size = SALT_SIZE + NONCE_SIZE + TAG_SIZE

    if len(payload) < minimum_size:
        raise ValueError("Encrypted payload is too short.")

    salt = payload[:SALT_SIZE]
    nonce = payload[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    ciphertext = payload[SALT_SIZE + NONCE_SIZE:]

    key = derive_key(password, salt)

    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except InvalidTag as exc:
        raise ValueError(
            "Decryption failed: incorrect password or corrupted data."
        ) from exc


def main():
    import zlib
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent
    secret = base / "data" / "secret.txt"
    cover = base / "data" / "cover.png"
    output = base / "output" / "stego.png"
    recovered = base / "output" / "recovered.txt"

    if not secret.is_file():
        raise FileNotFoundError(f"Secret file not found: {secret}")

    if not cover.is_file():
        raise FileNotFoundError(f"Cover image not found: {cover}")

    # 1. Read original file
    original_data = secret.read_bytes()
    print(f"Original size: {len(original_data):,} bytes")

    # 2. Compress
    compressed_data = zlib.compress(original_data, level=9)
    print(f"Compressed size: {len(compressed_data):,} bytes")

    # 3. Encrypt compressed data
    password = getpass.getpass("Enter encryption password: ")
    if not password:
        raise ValueError("Password cannot be empty.")

    encrypted_data = encrypt_payload(compressed_data, password)
    overhead = len(encrypted_data) - len(compressed_data)

    print(f"Encrypted payload: {len(encrypted_data):,} bytes")
    print(f"Encryption overhead: {overhead} bytes")

    # 4. Embed encrypted data in the image
    encode_image(cover, output, encrypted_data)
    print(f"Embedded payload into: {output}")

    # 5. Extract encrypted data
    extracted_data = decode_image(output)
    print(f"Extracted payload: {len(extracted_data):,} bytes")

    # 6. Simulate the receiver entering the password
    receiver_password = getpass.getpass("Enter decryption password: ")

    # 7. Decrypt, authenticate, and decompress
    decrypted_data = decrypt_payload(
        extracted_data,
        receiver_password,
    )
    recovered_data = zlib.decompress(decrypted_data)

    # 8. Verify exact recovery
    if recovered_data != original_data:
        raise RuntimeError(
            "Verification failed: recovered data differs from original."
        )

    recovered.parent.mkdir(parents=True, exist_ok=True)
    recovered.write_bytes(recovered_data)

    print(f"Recovered file: {recovered}")
    print("SUCCESS: original and recovered files match exactly.")


if __name__ == "__main__":
    main()
