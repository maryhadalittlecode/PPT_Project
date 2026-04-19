import argparse
import os
import time
import cv2

from canny_helpers import load_images, read_grayscale, compute_canny_stages, to_uint8
from hysteresis import METHODS


def run_one_image(image_path, output_dir, method_name, low, high, connectivity, save_intermediate=False):
    image = read_grayscale(image_path)
    stages = compute_canny_stages(image, low, high)

    hysteresis_fn = METHODS[method_name]

    start = time.perf_counter()
    final_edges = hysteresis_fn(stages["strong_map"], stages["weak_map"], connectivity)
    elapsed = time.perf_counter() - start

    base = os.path.splitext(os.path.basename(image_path))[0]
    image_out_dir = os.path.join(output_dir, base)
    os.makedirs(image_out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(image_out_dir, "final_edges.png"), to_uint8(final_edges))

    if save_intermediate:
        cv2.imwrite(os.path.join(image_out_dir, "input.png"), image)
        cv2.imwrite(os.path.join(image_out_dir, "blurred.png"), to_uint8(stages["blurred"]))
        cv2.imwrite(os.path.join(image_out_dir, "magnitude.png"), to_uint8(stages["magnitude"]))
        cv2.imwrite(os.path.join(image_out_dir, "nms.png"), to_uint8(stages["nms"]))
        cv2.imwrite(os.path.join(image_out_dir, "strong_map.png"), to_uint8(stages["strong_map"]))
        cv2.imwrite(os.path.join(image_out_dir, "weak_map.png"), to_uint8(stages["weak_map"]))

    print(f"{base}: {method_name} took {elapsed:.6f} seconds")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Simple project scaffold for parallel Canny hysteresis.")
    parser.add_argument("--data_dir", type=str, default="data", help="Folder with input images")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder for saved results")
    parser.add_argument("--method", type=str, default="sequential_bfs",
                        choices=["sequential_bfs", "frontier_parallel", "union_find"],
                        help="Which hysteresis method to use")
    parser.add_argument("--low", type=float, default=50.0, help="Low threshold")
    parser.add_argument("--high", type=float, default=100.0, help="High threshold")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8], help="Pixel connectivity")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate Canny results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = load_images(args.data_dir)

    if not image_paths:
        print(f"No images found in {args.data_dir}")
        return

    total = 0.0
    for image_path in image_paths:
        total += run_one_image(
            image_path=image_path,
            output_dir=args.output_dir,
            method_name=args.method,
            low=args.low,
            high=args.high,
            connectivity=args.connectivity,
            save_intermediate=args.save_intermediate,
        )

    print(f"Processed {len(image_paths)} image(s). Total hysteresis time: {total:.6f} seconds")


if __name__ == "__main__":
    main()
