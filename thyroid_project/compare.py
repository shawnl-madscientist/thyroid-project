import pandas as pd
import matplotlib
matplotlib.use("Agg")   # draw graphs into image files (no pop-up window)
import matplotlib.pyplot as plt


def get_subject(name):
    # Pull the animal's ID out of a photo name.
    # Example: "6_VhMC_M2_SP103.jpg" -> "SP103"
    name = str(name).strip().replace(".jpg", "")
    return name.split("_")[-1]


def compare_to_handscore(df, handscore_file, comparison_csv, comparison_plot):
    # Compare automatic areas (df, from run_batch) with hand-scored areas,
    # matched by subject and side. Saves a CSV and a graph, and returns the table.

    # Read the hand-scored areas (one row per lobe).
    hand = pd.read_excel(handscore_file, sheet_name="Sheet1")[["Image", "Side", "Area"]]
    hand = hand.rename(columns={"Area": "handscore_mm2"})
    hand["Subject"] = hand["Image"].apply(get_subject)
    hand["Side"] = hand["Side"].str.strip().str.upper()
    hand = hand[["Subject", "Side", "handscore_mm2"]]

    # Reshape the automatic results to one row per lobe, the same layout.
    auto = df.melt(id_vars="filename",
                   value_vars=["left_thyroid_mm2", "right_thyroid_mm2"],
                   var_name="Side", value_name="automated_mm2")
    auto["Side"] = auto["Side"].map({"left_thyroid_mm2": "L", "right_thyroid_mm2": "R"})
    auto["Subject"] = auto["filename"].apply(get_subject)
    auto = auto[["Subject", "Side", "automated_mm2"]]

    # Match by animal and side. how="left" keeps every hand-scored lobe.
    compare = hand.merge(auto, on=["Subject", "Side"], how="left")
    compare["difference_mm2"] = compare["automated_mm2"] - compare["handscore_mm2"]
    compare["percent_diff"] = compare["difference_mm2"] / compare["handscore_mm2"] * 100
    compare.to_csv(comparison_csv, index=False)

    # Only lobes with BOTH values can be compared.
    usable = compare.dropna(subset=["handscore_mm2", "automated_mm2"])
    print("\n-- Comparison with hand-scored areas --")
    print(f"Hand-scored lobes:             {len(compare)}")
    print(f"  with an automatic mm2 value: {compare['automated_mm2'].notna().sum()}")
    print(f"  usable for comparison:       {len(usable)}")

    if len(usable) >= 3:
        print(f"Average difference (auto - hand): {usable['difference_mm2'].mean():.4f} mm2")
        print(f"Average size of error:            {usable['difference_mm2'].abs().mean():.4f} mm2")
        print(f"Average size of error (percent):  {usable['percent_diff'].abs().mean():.1f} %")
        print(f"Correlation (Pearson):            {usable['automated_mm2'].corr(usable['handscore_mm2']):.3f}")
        print(f"Correlation (Spearman):           {usable['automated_mm2'].corr(usable['handscore_mm2'], method='spearman'):.3f}")

        # Side check: swap L and R. A clearly higher number than Pearson above
        # would mean the left/right labels may be reversed.
        swapped = auto.copy()
        swapped["Side"] = swapped["Side"].map({"L": "R", "R": "L"})
        swap_usable = hand.merge(swapped, on=["Subject", "Side"], how="left").dropna(
            subset=["handscore_mm2", "automated_mm2"])
        print(f"Correlation if left/right swapped: {swap_usable['automated_mm2'].corr(swap_usable['handscore_mm2']):.3f}")
    else:
        print("Too few lobes with both values to calculate summary numbers.")

    # Graph: each dot is one lobe; the dashed line means perfect agreement.
    if len(usable) > 0:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(usable["handscore_mm2"], usable["automated_mm2"], color="black", s=25)
        top = max(usable["handscore_mm2"].max(), usable["automated_mm2"].max()) * 1.05
        ax.plot([0, top], [0, top], linestyle="--", color="gray")
        ax.set_xlabel("Hand-scored area (mm2)")
        ax.set_ylabel("Automatic area (mm2)")
        ax.set_title("Automatic vs hand-scored thyroid area")
        plt.tight_layout()
        plt.savefig(comparison_plot, dpi=300)
        plt.close()

    print(f"\nComparison table saved to {comparison_csv}")
    print(f"Graph saved to {comparison_plot}")
    return compare