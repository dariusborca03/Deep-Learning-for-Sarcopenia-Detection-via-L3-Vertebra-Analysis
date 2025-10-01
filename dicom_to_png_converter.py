# dicom_debug_converter.py - Debug și fix pentru conversie DICOM
import os
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def debug_dicom_file(dicom_path):
    """Debug pentru a înțelege datele DICOM"""
    print(f"Analizez: {dicom_path}")

    dicom = pydicom.dcmread(dicom_path)
    img = dicom.pixel_array

    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Min value: {np.min(img)}")
    print(f"Max value: {np.max(img)}")
    print(f"Mean value: {np.mean(img)}")

    # Afișează histograma valorilor
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(img.flatten(), bins=100, alpha=0.7)
    plt.title("Histograma valorilor pixel")
    plt.xlabel("Valoare pixel")
    plt.ylabel("Frecventa")

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title("Imagine raw (fara windowing)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("debug_dicom.png")
    plt.show()

    return img


def convert_dicom_simple(dicom_path, output_path):
    """Conversie simplă fără windowing complex"""
    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array.astype(np.float32)

        # Metodă 1: Auto-contrast simplu
        img_min = np.min(img)
        img_max = np.max(img)

        if img_max > img_min:
            img_normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_normalized = np.zeros_like(img, dtype=np.uint8)

        # Salvează
        pil_image = Image.fromarray(img_normalized, mode='L')
        pil_image.save(output_path)

        return True, f"Range: {img_min:.1f} to {img_max:.1f}"

    except Exception as e:
        return False, str(e)


def convert_dicom_percentile(dicom_path, output_path, low_percentile=1, high_percentile=99):
    """Conversie cu percentile pentru windowing"""
    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array.astype(np.float32)

        # Windowing cu percentile
        p_low = np.percentile(img, low_percentile)
        p_high = np.percentile(img, high_percentile)

        img_clipped = np.clip(img, p_low, p_high)
        img_normalized = ((img_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

        # Salvează
        pil_image = Image.fromarray(img_normalized, mode='L')
        pil_image.save(output_path)

        return True, f"Percentile {low_percentile}%-{high_percentile}%: {p_low:.1f} to {p_high:.1f}"

    except Exception as e:
        return False, str(e)


def convert_dicom_ct_window(dicom_path, output_path):
    """Conversie cu windowing specific CT"""
    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array.astype(np.float32)

        # Încearcă să citească window din DICOM header
        try:
            window_center = float(dicom.WindowCenter[0] if hasattr(dicom, 'WindowCenter') else 40)
            window_width = float(dicom.WindowWidth[0] if hasattr(dicom, 'WindowWidth') else 400)
        except:
            # Windowing pentru abdomen/soft tissue
            window_center = 40
            window_width = 400

        # Aplică windowing
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        img_windowed = np.clip(img, img_min, img_max)
        img_normalized = ((img_windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # Salvează
        pil_image = Image.fromarray(img_normalized, mode='L')
        pil_image.save(output_path)

        return True, f"CT Window: C={window_center}, W={window_width}"

    except Exception as e:
        return False, str(e)


def test_conversion_methods():
    """Testează diferite metode de conversie pe primul fișier"""
    input_dir = "data/images/"

    # Găsește primul fișier DICOM
    dicom_files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    if not dicom_files:
        print("Nu s-au gasit fisiere DICOM!")
        return

    dicom_files.sort()
    test_file = dicom_files[0]
    dicom_path = os.path.join(input_dir, test_file)

    print(f"Testez conversiile pe: {test_file}")
    print("=" * 50)

    # Debug info
    debug_dicom_file(dicom_path)

    # Testează metodele
    methods = [
        ("simple", convert_dicom_simple),
        ("percentile", convert_dicom_percentile),
        ("ct_window", convert_dicom_ct_window)
    ]

    results = []

    for method_name, method_func in methods:
        output_path = f"test_{method_name}.png"
        success, info = method_func(dicom_path, output_path)

        if success:
            print(f"✓ {method_name}: {info}")
            results.append((method_name, output_path))
        else:
            print(f"✗ {method_name}: {info}")

    # Afișează rezultatele
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
        if len(results) == 1:
            axes = [axes]

        for i, (method_name, png_path) in enumerate(results):
            img = Image.open(png_path)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Metoda: {method_name}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig("conversion_comparison.png", dpi=150)
        plt.show()

        print(f"\nComparația metodelor salvată în: conversion_comparison.png")


def convert_all_with_best_method():
    """Convertește toate cu cea mai bună metodă"""
    input_dir = "data/images/"
    output_dir = "png_fixed/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dicom_files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    dicom_files.sort()

    print(f"Convertesc {len(dicom_files)} fisiere cu metoda percentile...")

    for i, filename in enumerate(dicom_files):
        dicom_path = os.path.join(input_dir, filename)
        png_path = os.path.join(output_dir, filename.replace('.dcm', '.png'))

        success, info = convert_dicom_percentile(dicom_path, png_path)

        if success:
            print(f"✓ {i + 1}/{len(dicom_files)}: {filename}")
        else:
            print(f"✗ {i + 1}/{len(dicom_files)}: {filename} - {info}")

    print(f"\nConversie completă! Imaginile sunt în: {output_dir}")


if __name__ == "__main__":
    print("DICOM Debug Converter")
    print("=" * 30)

    # Pasul 1: Testează metodele
    test_conversion_methods()

    # Pasul 2: Convertește toate cu metoda cea mai bună
    print("\n" + "=" * 50)
    convert_all_with_best_method()