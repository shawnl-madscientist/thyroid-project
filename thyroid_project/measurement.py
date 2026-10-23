# ══════════════════════════════════════════════════════════════════════════════
# What this script does (in plain words)
#
# For every photo in a folder, it:
#   1. Finds the ruler in the photo and works out how many pixels equal 1 mm.
#   2. Finds the two thyroid lobes by their colour and shape.
#   3. Measures the size of each lobe (in pixels, and in mm² if the ruler was found).
#   4. Saves a picture with the lobes outlined, so you can check the result by eye.
#   5. Saves all the measurements in one spreadsheet (CSV file).
#   6. Compares the automatic measurements (mm²) with the hand-scored areas,
#      matching them up by animal (subject) and side (left / right).
#
# A "pixel" is one tiny dot of a digital picture. A photo is a grid of pixels.
# ══════════════════════════════════════════════════════════════════════════════

# ── Tools (libraries) this script borrows ─────────────────────────────────────
import cv2              # OpenCV: reads, edits, measures and saves images
import numpy as np      # NumPy: fast maths on large grids of numbers (an image is one)
import os               # Works with files and folders (finding photos, joining paths)
import pandas as pd     # pandas: builds tables and saves them as spreadsheets
import matplotlib       # matplotlib: draws graphs
matplotlib.use("Agg")   # draw graphs straight into image files (no pop-up window needed)
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Work out the scale (how many pixels make 1 millimetre)
# ══════════════════════════════════════════════════════════════════════════════
def detect_pixels_per_mm(image):
    # Look at the ruler's tick marks. The distance between two neighbouring marks
    # is 1 mm, so measuring that gap in pixels gives us the scale.

    # Turn the colour photo into black-and-white, because we only need
    # light vs dark to spot the ruler marks.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Keep track of the best answer found so far.
    best_spacing = None      # gap between marks, in pixels (none found yet)
    best_peak_count = 0      # how many marks were seen when we found it

    # The ruler might lie sideways or stand upright in the photo,
    # so try both directions and keep whichever finds more marks.
    for orientation in ['horizontal', 'vertical']:

        # Mark every dark pixel (darker than 90 on a 0-255 scale) as 1, everything else as 0.
        dark = (gray < 90).astype(np.uint8)

        if orientation == 'horizontal':
            # Count the dark pixels in each row of the photo.
            row_sums = dark.sum(axis=1)
            # The row with the most dark pixels is most likely the ruler.
            ruler_row = np.argmax(row_sums)
            # Cut out a thin strip (10 pixels above and below that row) around the ruler.
            strip = dark[max(0, ruler_row - 10):ruler_row + 10, :]
            # Count dark pixels in each column of the strip.
            # The count goes up wherever there is a tick mark, and down in the gaps.
            profile = strip.sum(axis=0).astype(float)
        else:
            # Same idea, but for a ruler standing upright: swap rows and columns.
            col_sums = dark.sum(axis=0)
            ruler_col = np.argmax(col_sums)
            strip = dark[:, max(0, ruler_col - 10):ruler_col + 10]
            profile = strip.sum(axis=1).astype(float)

        # Smooth the counts: replace each value by the average of itself and its
        # 3 neighbours on each side (7 values in total). This removes small bumps
        # caused by noise, so each tick mark becomes one clean hump.
        kernel = np.ones(7) / 7
        profile = np.convolve(profile, kernel, mode='same')

        # A value counts as "on a tick mark" if it is above 25% of the highest value.
        threshold = profile.max() * 0.25

        # Walk along the counts from start to end and record where each tick mark is.
        in_peak = False      # are we currently on a tick mark?
        peaks = []           # the centre position of every tick mark found
        for i, val in enumerate(profile):
            if val > threshold and not in_peak:
                # The counts just rose above the threshold: a tick mark starts here.
                in_peak = True
                peak_start = i
            elif val <= threshold and in_peak:
                # The counts just dropped back down: the tick mark has ended.
                # Store its centre (halfway between its start and end).
                in_peak = False
                peaks.append((peak_start + i) // 2)

        # Only trust the result if at least 4 tick marks were found.
        if len(peaks) >= 4:
            # Distance (in pixels) between each tick mark and the next one.
            spacings = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
            # Ignore gaps that are unrealistically small or large (likely noise).
            spacings = [s for s in spacings if 5 < s < 500]
            if not spacings:
                continue     # nothing usable in this direction, try the other one

            # The "typical" gap is the middle value of all the gaps (the median).
            median_spacing = np.median(spacings)
            # If some gaps are clearly bigger than typical (more than 1.4 times),
            # use the typical size of those bigger gaps instead.
            large_spacings = [s for s in spacings if s > median_spacing * 1.4]
            if large_spacings:
                median_spacing = np.median(large_spacings)

            # Keep this answer only if it found more tick marks than the
            # other direction did.
            if len(peaks) > best_peak_count:
                best_spacing = median_spacing
                best_peak_count = len(peaks)

    # Pixels per mm (or None if no ruler could be read).
    return best_spacing


# ══════════════════════════════════════════════════════════════════════════════
# Find which row of the photo the ruler is in
# ══════════════════════════════════════════════════════════════════════════════
def get_ruler_row(image):
    # Black-and-white version of the photo.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Mark very dark pixels (darker than 80) as 1, everything else as 0.
    dark = (gray < 80).astype(np.uint8)
    # Count the dark pixels in each row.
    row_sums = dark.sum(axis=1)
    # The row with the most dark pixels is the ruler. We use this later to
    # hide the ruler so it is not mistaken for a thyroid.
    return int(np.argmax(row_sums))


# ══════════════════════════════════════════════════════════════════════════════
# Shape checks (used to tell a thyroid lobe apart from other coloured patches)
# ══════════════════════════════════════════════════════════════════════════════
def circularity(contour):
    # How round is the shape? A perfect circle scores 1.
    # Long, thin or very jagged shapes score close to 0.
    area = cv2.contourArea(contour)             # space inside the outline
    perimeter = cv2.arcLength(contour, True)    # length of the outline
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)


def solidity(contour):
    # How "filled in" is the shape? Imagine stretching a rubber band around it.
    # Score = shape's area / area inside the rubber band.
    # 1 means no dents or gaps; a low score means a lot of dents or a ragged edge.
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)              # the rubber-band outline
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Find the thyroid lobes in the photo
# ══════════════════════════════════════════════════════════════════════════════
def segment_thyroid(image, pixels_per_mm, ruler_row):
    # Switch the photo to a colour system that separates the colour itself
    # (shade) from how vivid and how bright it is. This makes it easier to say
    # "find pinkish-brown things" even when lighting changes a little.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build a black-and-white "mask": white wherever the colour is in our range
    # (pink / red / brown), black everywhere else.
    # Reds sit at both ends of the colour scale, so we use two ranges and combine them.
    mask1 = cv2.inRange(hsv, np.array([0,  8, 25]), np.array([40, 255, 240]))
    mask2 = cv2.inRange(hsv, np.array([165, 8, 25]), np.array([180, 255, 240]))
    mask = cv2.bitwise_or(mask1, mask2)

    # Paint over the ruler strip in black so the ruler is never picked up.
    if ruler_row is not None:
        mask[max(0, ruler_row - 8):ruler_row + 8, :] = 0

    # Tidy the mask using a small 5x5-pixel square:
    #   first pass removes tiny white specks,
    #   second pass fills tiny black holes inside the white patches.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Trace the outer edge of every white patch. Each traced edge is an "outline".
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Go through the outlines and keep only the ones that look like a thyroid lobe.
    valid_regions = []
    for cnt in contours:
        # Size of this patch in pixels.
        area_px = cv2.contourArea(cnt)

        # Too tiny to be a lobe: skip it.
        if area_px < 300:
            continue

        # If we know the scale, also check the real-life size (in mm²)
        # and skip anything too big or too small to be a thyroid lobe.
        if pixels_per_mm:
            area_mm2 = area_px / (pixels_per_mm ** 2)
            if area_mm2 > 2.0 or area_mm2 < 0.005:
                continue

        # Check the shape. These limits are loose on purpose, because the
        # lobes are bean- or oval-shaped (not perfectly round).
        circ = circularity(cnt)
        sol = solidity(cnt)
        if circ < 0.08 or sol < 0.3:
            continue

        # Find the centre point of the patch (cx = left-right, cy = up-down).
        # The "m00" value is the patch's size; if it is zero there is no patch.
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # This patch passed every check: keep its size, centre and outline.
        valid_regions.append((area_px, cx, cy, cnt))

    # Return the surviving patches and the mask (the mask is saved for checking).
    return valid_regions, mask


# ══════════════════════════════════════════════════════════════════════════════
# STEPS 3 and 4: Measure one photo and save a picture to check it
# ══════════════════════════════════════════════════════════════════════════════
def measure_and_visualize(image_path, output_dir):
    # Open the photo. If it cannot be opened, report "nothing" for every result.
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None, None, None

    # Run the steps above on this photo.
    pixels_per_mm = detect_pixels_per_mm(image)                      # the scale
    ruler_row = get_ruler_row(image)                                 # where the ruler is
    regions, mask = segment_thyroid(image, pixels_per_mm, ruler_row) # candidate lobes

    # Sort from biggest to smallest and keep the two biggest (the two lobes).
    regions.sort(reverse=True)
    top2 = regions[:2]

    # Start with "nothing found" for each lobe.
    right_px = left_px = 0
    right_cnt = left_cnt = None
    # The middle of the photo, left to right.
    img_mid = image.shape[1] / 2

    if len(top2) == 1:
        # Only one lobe found: decide which side it is by whether it sits in the
        # left half or the right half of the photo.
        # (This code labels the lobe on the LEFT of the photo as "right",
        # and the lobe on the RIGHT of the photo as "left".)
        area_px, cx, cy, cnt = top2[0]
        if cx < img_mid:
            right_px, right_cnt = area_px, cnt
        else:
            left_px, left_cnt = area_px, cnt
    elif len(top2) >= 2:
        # Two lobes found: sort them by horizontal position.
        # The one furthest left in the photo is labelled "right", the other "left".
        sorted_x = sorted(top2, key=lambda r: r[1])
        right_px, right_cnt = sorted_x[0][0], sorted_x[0][3]
        left_px, left_cnt = sorted_x[1][0], sorted_x[1][3]

    # Draw the outlines on a copy of the photo so you can check the result.
    # (Colours are given as Blue, Green, Red: (0, 255, 255) is yellow, (0, 255, 0) is green.)
    overlay = image.copy()
    if right_cnt is not None:
        # Yellow outline around the "right" lobe.
        cv2.drawContours(overlay, [right_cnt], -1, (0, 255, 255), 3)
        # Find the lobe's centre and write the letter "R" there.
        M = cv2.moments(right_cnt)
        if M["m00"] > 0:
            cx2, cy2 = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(overlay, "R", (cx2 - 15, cy2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    if left_cnt is not None:
        # Green outline around the "left" lobe, with the letter "L" at its centre.
        cv2.drawContours(overlay, [left_cnt], -1, (0, 255, 0), 3)
        M = cv2.moments(left_cnt)
        if M["m00"] > 0:
            cx2, cy2 = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(overlay, "L", (cx2 - 15, cy2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Save two pictures per photo, named after the original file:
    #   *_overlay.jpg = the photo with the lobes outlined
    #   *_mask.jpg    = the black-and-white colour mask
    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base}_overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(output_dir, f"{base}_mask.jpg"), mask)

    def to_mm2(px):
        # Convert an area from pixels to square millimetres.
        # One mm is `pixels_per_mm` pixels wide, so one mm² is that number squared.
        # If there is no scale (or no lobe), give back "nothing" instead.
        if pixels_per_mm and px > 0:
            return round(px / (pixels_per_mm ** 2), 4)
        return None

    # Add the two lobes together, then return all the results for this photo:
    # total, right and left, each as pixels and as mm².
    total_px = right_px + left_px
    return total_px, to_mm2(total_px), right_px, to_mm2(right_px), left_px, to_mm2(left_px)


def run_batch(folder, overlay_dir, output_csv):
    os.makedirs(overlay_dir, exist_ok=True)
    results = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg"):
            path = os.path.join(folder, filename)
            total_px, total_mm2, r_px, r_mm2, l_px, l_mm2 = measure_and_visualize(path, overlay_dir)
            results.append({
                "filename": filename,
                "total_area_mm2": total_mm2,
                "right_thyroid_mm2": r_mm2,
                "left_thyroid_mm2": l_mm2,
                "total_area_pixels": total_px,
                "right_thyroid_pixels": r_px,
                "left_thyroid_pixels": l_px
            })
            if r_mm2 is not None or l_mm2 is not None:
                print(f"{filename}: Right={r_mm2} mm2 | Left={l_mm2} mm2")
            else:
                print(f"{filename}: no scale | Right={r_px}px Left={l_px}px")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df