import img
import denoise
import rotation
import compression
import aberration


def main():
    initial = img.complete()
    no_aberration = aberration.fix(initial)
    rotated = rotation.fix(no_aberration)
    no_noise = denoise.fix(rotated)
    compressed = compression.compress_image(no_noise)

    steps = {
        "initial": initial,
        "no-aberration": no_aberration,
        "rotated": rotated,
        "denoised": no_noise,
        "compressed": compressed,
    }
    for name, data in steps.items():
        img.save(f"complete-{name}.png", data)


if __name__ == "__main__":
    main()
