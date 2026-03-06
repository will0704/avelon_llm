"""
Generate synthetic training data for the 4-class document classifier.

Creates synthetic images for classes that lack real training data:
- government_id  → copies existing IDVerifier/validID images
- not_id         → copies existing IDVerifier/nonValid images
- proof_of_income  → generates synthetic payslip-like images
- proof_of_address → generates synthetic utility-bill-like images

The output structure matches ImageFolder convention:
    data/classifier/train/<class_name>/
    data/classifier/val/<class_name>/

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --count 30
"""
import os
import sys
import argparse
import random
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--count", type=int, default=25,
        help="Number of synthetic images per class (train set). Val set gets count//3."
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(ROOT_DIR, "data", "classifier"),
        help="Output directory"
    )
    parser.add_argument(
        "--id-source", type=str,
        default=os.path.join(ROOT_DIR, "..", "resources_online", "IDVerifier", "data"),
        help="Path to existing IDVerifier dataset"
    )
    return parser.parse_args()


def _try_get_font(size: int = 16):
    """Try to get a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def _random_color(base: tuple, variance: int = 30) -> tuple:
    """Generate a color near a base color."""
    return tuple(
        max(0, min(255, c + random.randint(-variance, variance)))
        for c in base
    )


def generate_payslip_image(idx: int) -> Image.Image:
    """Generate a synthetic payslip / pay stub image."""
    width, height = random.choice([(600, 400), (500, 350), (700, 500)])
    bg_color = _random_color((245, 245, 240), 10)
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    font_title = _try_get_font(20)
    font_body = _try_get_font(14)
    font_small = _try_get_font(11)

    # Company header
    companies = [
        "ABC Corporation", "PhilTech Inc.", "Metro Services Ltd.",
        "Global Outsourcing PH", "Manila Trading Co.", "PH Solutions Corp.",
        "Pacific Holdings Inc.", "Island Enterprises", "Star Computing Ltd."
    ]
    company = random.choice(companies)
    draw.text((20, 15), company, fill=(0, 0, 120), font=font_title)
    draw.text((20, 45), "PAYSLIP", fill=(0, 0, 0), font=font_title)

    # Horizontal line
    draw.line([(15, 75), (width - 15, 75)], fill=(0, 0, 0), width=2)

    # Employee info
    names = ["Juan Dela Cruz", "Maria Santos", "Pedro Reyes", "Ana Garcia",
             "Jose Mendoza", "Rosa Villanueva", "Carlos Torres", "Elena Cruz"]
    name = random.choice(names)
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    month = random.choice(months)
    year = random.choice([2023, 2024, 2025])

    draw.text((20, 85), f"Employee: {name}", fill=(0, 0, 0), font=font_body)
    draw.text((20, 105), f"Period: {month} {year}", fill=(0, 0, 0), font=font_body)
    draw.text((20, 125), f"ID: EMP-{random.randint(1000, 9999)}", fill=(0, 0, 0), font=font_body)

    # Earnings table
    y = 160
    draw.text((20, y), "EARNINGS", fill=(0, 0, 100), font=font_body)
    y += 25
    basic = random.randint(15000, 80000)
    allowance = random.randint(1000, 10000)
    overtime = random.randint(0, 5000)
    gross = basic + allowance + overtime

    for label, amount in [("Basic Salary", basic), ("Allowance", allowance), ("Overtime", overtime)]:
        draw.text((30, y), label, fill=(0, 0, 0), font=font_small)
        draw.text((width - 150, y), f"PHP {amount:,.2f}", fill=(0, 0, 0), font=font_small)
        y += 20

    draw.line([(20, y), (width - 20, y)], fill=(100, 100, 100), width=1)
    y += 5
    draw.text((30, y), "GROSS PAY", fill=(0, 0, 0), font=font_body)
    draw.text((width - 150, y), f"PHP {gross:,.2f}", fill=(0, 0, 100), font=font_body)
    y += 30

    # Deductions
    draw.text((20, y), "DEDUCTIONS", fill=(100, 0, 0), font=font_body)
    y += 25
    sss = round(gross * 0.045, 2)
    philhealth = round(gross * 0.025, 2)
    tax = round(gross * random.uniform(0.05, 0.15), 2)
    total_ded = sss + philhealth + tax

    for label, amount in [("SSS", sss), ("PhilHealth", philhealth), ("Tax", tax)]:
        draw.text((30, y), label, fill=(0, 0, 0), font=font_small)
        draw.text((width - 150, y), f"PHP {amount:,.2f}", fill=(150, 0, 0), font=font_small)
        y += 20

    draw.line([(20, y), (width - 20, y)], fill=(0, 0, 0), width=2)
    y += 8
    net = gross - total_ded
    draw.text((30, y), "NET PAY", fill=(0, 0, 0), font=font_title)
    draw.text((width - 180, y), f"PHP {net:,.2f}", fill=(0, 100, 0), font=font_title)

    return img


def generate_utility_bill_image(idx: int) -> Image.Image:
    """Generate a synthetic utility bill image."""
    width, height = random.choice([(550, 700), (600, 750), (500, 650)])
    bg_color = _random_color((250, 250, 245), 5)
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    font_title = _try_get_font(22)
    font_body = _try_get_font(14)
    font_small = _try_get_font(11)

    # Utility company header
    utilities = [
        ("Meralco", "Electric Bill"),
        ("Manila Water", "Water Bill"),
        ("Maynilad", "Water Bill"),
        ("PLDT", "Internet / Phone Bill"),
        ("Globe Telecom", "Internet Bill"),
        ("Converge ICT", "Fiber Internet Bill"),
    ]
    company, bill_type = random.choice(utilities)

    draw.text((20, 15), company, fill=(0, 50, 150), font=font_title)
    draw.text((20, 45), bill_type.upper(), fill=(80, 80, 80), font=font_body)

    # Border
    draw.rectangle([(10, 10), (width - 10, height - 10)], outline=(0, 0, 0), width=1)
    draw.line([(10, 75), (width - 10, 75)], fill=(0, 0, 0), width=2)

    # Account info
    y = 90
    names = ["Juan Dela Cruz", "Maria Santos", "Pedro Reyes", "Ana Garcia",
             "Jose Mendoza", "Carlos Torres", "Elena Cruz"]
    name = random.choice(names)
    account_no = f"{random.randint(100, 999)}-{random.randint(1000, 9999)}-{random.randint(100, 999)}"

    addresses = [
        "123 Rizal St., Brgy. San Antonio, Makati City",
        "45 Mabini Ave., Brgy. Poblacion, Quezon City",
        "789 Bonifacio Rd., Brgy. Commonwealth, Manila",
        "12 Luna St., Brgy. Sta. Cruz, Pasig City",
        "34 Del Pilar Blvd., Brgy. Kapitolyo, Mandaluyong",
        "56 Aguinaldo Dr., Brgy. Daang Hari, Las Piñas",
    ]
    address = random.choice(addresses)

    draw.text((20, y), f"Account Name: {name}", fill=(0, 0, 0), font=font_body)
    y += 22
    draw.text((20, y), f"Account No: {account_no}", fill=(0, 0, 0), font=font_body)
    y += 22
    draw.text((20, y), f"Service Address:", fill=(80, 80, 80), font=font_small)
    y += 18
    draw.text((30, y), address, fill=(0, 0, 0), font=font_small)
    y += 30

    # Bill period
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    month = random.choice(months)
    year = random.choice([2024, 2025])
    draw.text((20, y), f"Billing Period: {month} 1-30, {year}", fill=(0, 0, 0), font=font_body)
    y += 22
    draw.text((20, y), f"Due Date: {month} 15, {year}", fill=(150, 0, 0), font=font_body)
    y += 35

    # Usage details
    draw.line([(15, y), (width - 15, y)], fill=(0, 0, 0), width=1)
    y += 10
    draw.text((20, y), "CONSUMPTION DETAILS", fill=(0, 50, 150), font=font_body)
    y += 25

    if "Electric" in bill_type:
        kwh = random.randint(80, 500)
        rate = round(random.uniform(9.5, 12.5), 4)
        draw.text((30, y), f"kWh Consumed: {kwh}", fill=(0, 0, 0), font=font_small)
        y += 18
        draw.text((30, y), f"Rate per kWh: PHP {rate}", fill=(0, 0, 0), font=font_small)
        amount = round(kwh * rate, 2)
    elif "Water" in bill_type:
        cu_m = random.randint(10, 60)
        draw.text((30, y), f"Cu.M. Consumed: {cu_m}", fill=(0, 0, 0), font=font_small)
        amount = round(cu_m * random.uniform(20, 40), 2)
    else:
        plan = random.choice([1299, 1699, 1999, 2499, 2999])
        draw.text((30, y), f"Plan: PHP {plan}/mo", fill=(0, 0, 0), font=font_small)
        amount = float(plan)

    y += 30
    draw.line([(15, y), (width - 15, y)], fill=(0, 0, 0), width=1)
    y += 10
    vat = round(amount * 0.12, 2)
    total = round(amount + vat, 2)

    draw.text((30, y), "Subtotal", fill=(0, 0, 0), font=font_body)
    draw.text((width - 160, y), f"PHP {amount:,.2f}", fill=(0, 0, 0), font=font_body)
    y += 22
    draw.text((30, y), "VAT (12%)", fill=(0, 0, 0), font=font_small)
    draw.text((width - 160, y), f"PHP {vat:,.2f}", fill=(0, 0, 0), font=font_small)
    y += 25
    draw.line([(15, y), (width - 15, y)], fill=(0, 0, 0), width=2)
    y += 8
    draw.text((30, y), "TOTAL AMOUNT DUE", fill=(0, 0, 0), font=font_title)
    draw.text((width - 180, y), f"PHP {total:,.2f}", fill=(200, 0, 0), font=font_title)

    return img


def copy_existing_images(src_dir: str, dst_dir: str, max_count: int):
    """Copy images from src to dst, augmenting with flips if needed."""
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_dir):
        print(f"  Warning: Source not found: {src_dir}")
        return

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = [f for f in os.listdir(src_dir)
             if os.path.splitext(f)[1].lower() in extensions]

    copied = 0
    # First pass: direct copies
    for f in files:
        if copied >= max_count:
            break
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f"orig_{f}"))
        copied += 1

    # Second pass: augmented copies if we need more
    aug_idx = 0
    while copied < max_count and files:
        f = files[aug_idx % len(files)]
        try:
            img = Image.open(os.path.join(src_dir, f))
            # Random augmentation
            aug_type = random.choice(['flip', 'rotate', 'bright'])
            if aug_type == 'flip':
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif aug_type == 'rotate':
                img = img.rotate(random.randint(-15, 15), expand=True, fillcolor=(255, 255, 255))
            elif aug_type == 'bright':
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.7, 1.3))

            out_name = f"aug_{aug_idx}_{os.path.splitext(f)[0]}.png"
            img.save(os.path.join(dst_dir, out_name), "PNG")
            copied += 1
        except Exception as e:
            print(f"  Warning: augmentation failed for {f}: {e}")
        aug_idx += 1
        if aug_idx > max_count * 3:
            break

    print(f"  {dst_dir}: {copied} images")


def generate_synthetic_class(dst_dir: str, generator_fn, count: int):
    """Generate synthetic images for a class."""
    os.makedirs(dst_dir, exist_ok=True)
    for i in range(count):
        img = generator_fn(i)
        # Random resize for variety
        target_size = random.choice([(224, 224), (256, 256), (300, 300), (400, 300)])
        img = img.resize(target_size, Image.LANCZOS)
        img.save(os.path.join(dst_dir, f"synth_{i:04d}.png"), "PNG")
    print(f"  {dst_dir}: {count} images")


def main():
    args = get_args()
    train_count = args.count
    val_count = max(train_count // 3, 5)

    output = os.path.abspath(args.output)
    id_source = os.path.abspath(args.id_source)

    train_dir = os.path.join(output, "train")
    val_dir = os.path.join(output, "val")

    print(f"Output directory: {output}")
    print(f"Train count per class: {train_count}")
    print(f"Val count per class: {val_count}\n")

    # 1. government_id — copy from IDVerifier/validID
    print("=== government_id ===")
    copy_existing_images(
        os.path.join(id_source, "train", "validID"),
        os.path.join(train_dir, "government_id"),
        train_count
    )
    copy_existing_images(
        os.path.join(id_source, "val", "validID"),
        os.path.join(val_dir, "government_id"),
        val_count
    )

    # 2. not_id — copy from IDVerifier/nonValid
    print("\n=== not_id ===")
    copy_existing_images(
        os.path.join(id_source, "train", "nonValid"),
        os.path.join(train_dir, "not_id"),
        train_count
    )
    copy_existing_images(
        os.path.join(id_source, "val", "nonValid"),
        os.path.join(val_dir, "not_id"),
        val_count
    )

    # 3. proof_of_income — synthetic payslips
    print("\n=== proof_of_income ===")
    generate_synthetic_class(
        os.path.join(train_dir, "proof_of_income"),
        generate_payslip_image,
        train_count
    )
    generate_synthetic_class(
        os.path.join(val_dir, "proof_of_income"),
        generate_payslip_image,
        val_count
    )

    # 4. proof_of_address — synthetic utility bills
    print("\n=== proof_of_address ===")
    generate_synthetic_class(
        os.path.join(train_dir, "proof_of_address"),
        generate_utility_bill_image,
        train_count
    )
    generate_synthetic_class(
        os.path.join(val_dir, "proof_of_address"),
        generate_utility_bill_image,
        val_count
    )

    print(f"\nDone! Dataset ready at: {output}")
    print(f"Train with: python train_classifier.py --data-dir {output}")


if __name__ == "__main__":
    main()
