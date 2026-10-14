from steganography import encode_image, decode_image

encode_image(
    "data/cover.png",
    "output/stego.png",
    b"This is my secret multimedia project."
)

result = decode_image(
    "output/stego.png",
    len(b"This is my secret multimedia project.")
)

print(result.decode())