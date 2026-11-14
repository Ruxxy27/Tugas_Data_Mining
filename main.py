from mlxtend.frequent_patterns import fpgrowth
import time
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# TAHAP 1: MEMBACA DATASET DOSEN
# ==========================================

os.makedirs("output", exist_ok=True)

transactions_df = pd.read_csv(
    "dataset/transactions_arm_praktikum.csv"
)

df = pd.read_csv(
    "dataset/transactions_arm_praktikum_onehot.csv"
)

# Mengubah data transaksi menjadi list
transactions = [
    [item.strip() for item in row.split(",")]
    for row in transactions_df["Items"]
]

print("=" * 50)
print("PRAKTIKUM ASSOCIATION RULE MINING")
print("=" * 50)

print("Jumlah transaksi:", len(transactions))
print("Jumlah jenis item:", df.shape[1])
print("Ukuran data One-Hot:", df.shape)

print("\nLima transaksi pertama:")

for i, row in transactions_df.head().iterrows():
    print(row["TID"], ":", row["Items"])

# ==========================================
# TAHAP 2: VALIDASI ONE-HOT ENCODING
# ==========================================

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df_hasil = pd.DataFrame(
    te_array,
    columns=te.columns_
)

# Memeriksa kesesuaian dengan file dosen
sama = (
    df_hasil[df.columns].reset_index(drop=True)
    .equals(df.astype(bool).reset_index(drop=True))
)

print("\nValidasi One-Hot Encoding:", sama)

# ==========================================
# TAHAP 3: FREKUENSI ITEM
# ==========================================

frekuensi = df.sum().sort_values(ascending=False)

print("\nFrekuensi Kemunculan Item:")
print(frekuensi)

# Membuat grafik
plt.figure(figsize=(10, 6))

frekuensi.plot(
    kind="bar",
    color="skyblue",
    edgecolor="black"
)

plt.title("Frekuensi Kemunculan Item")
plt.xlabel("Nama Item")
plt.ylabel("Jumlah Kemunculan")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("output/grafik_frekuensi.png")
plt.show()

print("\nGrafik berhasil disimpan!")

# ==========================================
# 2. PROSES APRIORI
# ==========================================

print("\n" + "=" * 50)
print("PROSES APRIORI")
print("=" * 50)

# Mengubah data menjadi boolean
df = df.astype(bool)

# Menentukan frequent itemsets
frequent_itemsets = apriori(
    df,
    min_support=0.15,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nJumlah Frequent Itemsets:", len(frequent_itemsets))

# Menyimpan frequent itemsets
frequent_itemsets.to_csv(
    "output/frequent_itemsets.csv",
    index=False
)

# ==========================================
# 3. ASSOCIATION RULE MINING
# ==========================================

print("\n" + "=" * 50)
print("ASSOCIATION RULE MINING")
print("=" * 50)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

# Mengurutkan berdasarkan lift tertinggi
rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("\nHasil Association Rules:")
print(
    rules[
        ["antecedents", "consequents",
         "support", "confidence", "lift"]
    ]
)

print("\nJumlah Association Rules:", len(rules))

# Mengubah antecedents dan consequents
# agar mudah dibaca di file CSV
rules["antecedents"] = rules["antecedents"].apply(
    lambda x: ", ".join(list(x))
)

rules["consequents"] = rules["consequents"].apply(
    lambda x: ", ".join(list(x))
)

# Menyimpan hasil rules
rules.to_csv(
    "output/rules.csv",
    index=False
)

print("\nFile berhasil disimpan:")
print("- output/frequent_itemsets.csv")
print("- output/rules.csv")

# ==========================================
# 4. TUGAS 1 - EKSPERIMEN SUPPORT DAN CONFIDENCE
# ==========================================

print("\n" + "=" * 60)
print("TUGAS 1: EKSPERIMEN SUPPORT DAN CONFIDENCE")
print("=" * 60)

# A. Perbandingan min_support 0.10 dan 0.15
hasil_perbandingan = []

for support in [0.10, 0.15]:

    # Mencari frequent itemsets
    itemsets = apriori(
        df,
        min_support=support,
        use_colnames=True
    )

    # Menghasilkan rules dengan confidence minimal 0.70
    rules_uji = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=0.70
    )

    hasil_perbandingan.append({
        "Minimum Support": support,
        "Jumlah Frequent Itemsets": len(itemsets),
        "Jumlah Rules Confidence >= 0.70": len(rules_uji)
    })

    print(f"\nMinimum Support: {support}")
    print("Jumlah Frequent Itemsets:", len(itemsets))
    print("Jumlah Rules Confidence >= 0.70:", len(rules_uji))

    # Simpan hasil masing-masing support
    rules_uji.to_csv(
        f"output/rules_support_{str(support).replace('.', '_')}.csv",
        index=False
    )

# Simpan tabel perbandingan
df_perbandingan = pd.DataFrame(hasil_perbandingan)

print("\nTABEL PERBANDINGAN SUPPORT:")
print(df_perbandingan)

df_perbandingan.to_csv(
    "output/perbandingan_support.csv",
    index=False
)


# B. Mencari rules yang gagal pada confidence 0.70
print("\n" + "=" * 60)
print("RULES YANG TIDAK LOLOS CONFIDENCE 0.70")
print("=" * 60)

itemsets_015 = apriori(
    df,
    min_support=0.15,
    use_colnames=True
)

rules_dasar = association_rules(
    itemsets_015,
    metric="confidence",
    min_threshold=0.50
)

rules_gagal = rules_dasar[
    rules_dasar["confidence"] < 0.70
].copy()

rules_gagal["antecedents"] = rules_gagal["antecedents"].apply(
    lambda x: ", ".join(sorted(x))
)

rules_gagal["consequents"] = rules_gagal["consequents"].apply(
    lambda x: ", ".join(sorted(x))
)

print("\nJumlah rules yang tidak lolos:", len(rules_gagal))

print(
    rules_gagal[
        ["antecedents", "consequents", "support",
         "confidence", "lift"]
    ]
)

rules_gagal.to_csv(
    "output/rules_gagal_confidence.csv",
    index=False
)


# C. Pengujian berbagai nilai confidence
print("\n" + "=" * 60)
print("PENGUJIAN BERBAGAI MINIMUM CONFIDENCE")
print("=" * 60)

nilai_confidence = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

hasil_confidence = []

for confidence in nilai_confidence:

    rules_uji = association_rules(
        itemsets_015,
        metric="confidence",
        min_threshold=confidence
    )

    hasil_confidence.append({
        "Minimum Confidence": confidence,
        "Jumlah Rules": len(rules_uji)
    })

    print(
        f"Confidence {confidence}: "
        f"{len(rules_uji)} rules"
    )

df_confidence = pd.DataFrame(hasil_confidence)

print("\nTABEL HASIL CONFIDENCE:")
print(df_confidence)

df_confidence.to_csv(
    "output/perbandingan_confidence.csv",
    index=False
)

# Grafik perbandingan confidence
plt.figure(figsize=(9, 5))

plt.plot(
    df_confidence["Minimum Confidence"],
    df_confidence["Jumlah Rules"],
    marker="o",
    linewidth=2
)

plt.title("Pengaruh Minimum Confidence terhadap Jumlah Rules")
plt.xlabel("Minimum Confidence")
plt.ylabel("Jumlah Association Rules")
plt.xticks(nilai_confidence)
plt.grid(True)
plt.tight_layout()

plt.savefig("output/grafik_perbandingan_confidence.png")
plt.show()

print("\nEksperimen Tugas 1 selesai!")

# ==========================================
# 5. TUGAS 2 - ANALISIS MENDALAM ATURAN
# ==========================================

print("\n" + "=" * 60)
print("TUGAS 2: ANALISIS MENDALAM ATURAN")
print("=" * 60)

aturan_pilihan = [
    (
        frozenset(["Mie Instan"]),
        frozenset(["Telur"])
    ),
    (
        frozenset(["Susu"]),
        frozenset(["Roti"])
    )
]

hasil_analisis = []

for antecedent, consequent in aturan_pilihan:

    # Menghitung support X, Y, dan X gabungan Y
    support_x = df[list(antecedent)].all(axis=1).mean()
    support_y = df[list(consequent)].all(axis=1).mean()

    support_xy = df[
        list(antecedent | consequent)
    ].all(axis=1).mean()

    # Menghitung confidence dan lift
    confidence = support_xy / support_x
    lift = confidence / support_y

    # Menghitung leverage
    leverage = support_xy - (support_x * support_y)

    # Menghitung conviction
    conviction = (1 - support_y) / (1 - confidence)

    hasil_analisis.append({
        "Antecedents": ", ".join(sorted(antecedent)),
        "Consequents": ", ".join(sorted(consequent)),
        "Support": support_xy,
        "Confidence": confidence,
        "Lift": lift,
        "Leverage": leverage,
        "Conviction": conviction
    })

df_analisis = pd.DataFrame(hasil_analisis)

print("\nHASIL PERHITUNGAN:")
print(df_analisis.round(4).to_string(index=False))

df_analisis.to_csv(
    "output/analisis_tugas_2.csv",
    index=False
)

print("\nHasil analisis Tugas 2 berhasil disimpan!")

# ==========================================
# 6. TUGAS 3 - APRIORI VS FP-GROWTH
# ==========================================

print("\n" + "=" * 60)
print("TUGAS 3: PERBANDINGAN APRIORI VS FP-GROWTH")
print("=" * 60)

nilai_support = [0.15, 0.05]

hasil_perbandingan_algoritma = []

for support in nilai_support:

    print(f"\nMinimum Support: {support}")

    # APRIORI
    mulai = time.perf_counter()

    hasil_apriori = apriori(
        df,
        min_support=support,
        use_colnames=True
    )

    waktu_apriori = time.perf_counter() - mulai

    # FP-GROWTH
    mulai = time.perf_counter()

    hasil_fpgrowth = fpgrowth(
        df,
        min_support=support,
        use_colnames=True
    )

    waktu_fpgrowth = time.perf_counter() - mulai

    # Menampilkan hasil
    print("\nAPRIORI")
    print("Jumlah itemsets:", len(hasil_apriori))
    print(f"Waktu eksekusi: {waktu_apriori * 1000:.4f} ms")

    print("\nFP-GROWTH")
    print("Jumlah itemsets:", len(hasil_fpgrowth))
    print(f"Waktu eksekusi: {waktu_fpgrowth * 1000:.4f} ms")

    # Memastikan kedua algoritma menghasilkan itemsets sama
    itemsets_apriori = {
        (frozenset(row["itemsets"]), round(row["support"], 10))
        for _, row in hasil_apriori.iterrows()
    }

    itemsets_fp = {
        (frozenset(row["itemsets"]), round(row["support"], 10))
        for _, row in hasil_fpgrowth.iterrows()
    }

    print("\nHasil itemsets sama:", itemsets_apriori == itemsets_fp)

    hasil_perbandingan_algoritma.append({
        "Minimum Support": support,
        "Algoritma": "Apriori",
        "Jumlah Itemsets": len(hasil_apriori),
        "Waktu (ms)": waktu_apriori * 1000
    })

    hasil_perbandingan_algoritma.append({
        "Minimum Support": support,
        "Algoritma": "FP-Growth",
        "Jumlah Itemsets": len(hasil_fpgrowth),
        "Waktu (ms)": waktu_fpgrowth * 1000
    })


# Membuat tabel perbandingan
df_algoritma = pd.DataFrame(hasil_perbandingan_algoritma)

print("\n" + "=" * 60)
print("TABEL PERBANDINGAN ALGORITMA")
print("=" * 60)

print(df_algoritma.round(4).to_string(index=False))

df_algoritma.to_csv(
    "output/perbandingan_apriori_fpgrowth.csv",
    index=False
)

print("\nHasil perbandingan berhasil disimpan!")


# ==========================================
# 7. TUGAS 4 - STUDI KASUS MANDIRI
# TOKO RUXXY
# ==========================================

print("\n" + "=" * 60)
print("TUGAS 4: STUDI KASUS MANDIRI TOKO RUXXY")
print("=" * 60)

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt
import os

# Membuat folder dataset dan output jika belum tersedia
os.makedirs("dataset", exist_ok=True)
os.makedirs("output", exist_ok=True)


# ==========================================
# A. MEMBUAT DATASET TRANSAKSI
# ==========================================

print("\nA. DATASET TRANSAKSI TOKO RUXXY")

transactions4 = []

# Produk kebutuhan sarapan
transactions4 += [['Roti', 'Mentega', 'Susu', 'Telur']] * 8
transactions4 += [['Roti', 'Mentega', 'Susu']] * 6
transactions4 += [['Roti', 'Susu', 'Telur']] * 4
transactions4 += [['Roti', 'Mentega']] * 3

# Produk minuman
transactions4 += [['Susu', 'Kopi', 'Gula']] * 5
transactions4 += [['Kopi', 'Gula']] * 3

# Produk makanan instan
transactions4 += [['Mie Instan', 'Kecap', 'Telur']] * 6
transactions4 += [['Mie Instan', 'Kecap']] * 4

# Produk kebutuhan mandi
transactions4 += [['Sabun', 'Shampo']] * 6
transactions4 += [['Sabun']] * 3
transactions4 += [['Shampo']] * 2

# Membuat DataFrame transaksi
df_transaksi4 = pd.DataFrame({
    "TID": [
        f"TR{i+1}" for i in range(len(transactions4))
    ],
    "Items": [
        ", ".join(transaksi)
        for transaksi in transactions4
    ]
})

# Menyimpan dataset ke CSV
path_dataset4 = "dataset/transactions_toko_ruxxy.csv"

df_transaksi4.to_csv(
    path_dataset4,
    index=False
)

print("Dataset berhasil dibuat!")
print("Jumlah transaksi:", len(transactions4))
print("Jumlah jenis item:", len(set(
    item for transaksi in transactions4
    for item in transaksi
)))

print("\nLima transaksi pertama:")
print(df_transaksi4.head())


# ==========================================
# B. MEMBACA DATASET CSV
# ==========================================

print("\nB. MEMBACA DATASET CSV")

transactions_df4 = pd.read_csv(path_dataset4)

transactions4 = [
    [item.strip() for item in row.split(",")]
    for row in transactions_df4["Items"]
]

print("Dataset berhasil dibaca!")
print("Total transaksi:", len(transactions4))


# ==========================================
# C. ONE-HOT ENCODING
# ==========================================

print("\nC. PROSES ONE-HOT ENCODING")

te4 = TransactionEncoder()

te_array4 = te4.fit(transactions4).transform(transactions4)

df4 = pd.DataFrame(
    te_array4,
    columns=te4.columns_
)

print("Ukuran matriks One-Hot:", df4.shape)

print("\nHasil One-Hot Encoding:")
print(df4.head(10))

# Menyimpan hasil One-Hot Encoding
df4.to_csv(
    "dataset/transactions_toko_ruxxy_onehot.csv",
    index=False
)


# ==========================================
# D. MENENTUKAN FREQUENT ITEMSETS
# ==========================================

print("\nD. PROSES ALGORITMA APRIORI")

MIN_SUPPORT4 = 0.10

freq4 = apriori(
    df4,
    min_support=MIN_SUPPORT4,
    use_colnames=True
)

freq4 = freq4.sort_values(
    by="support",
    ascending=False
)

print("\nFrequent Itemsets:")

print(freq4.to_string(index=False))

print("\nMinimum Support:", MIN_SUPPORT4)
print("Jumlah Frequent Itemsets:", len(freq4))

# Menyimpan frequent itemsets
freq4.to_csv(
    "output/frequent_itemsets_toko_ruxxy.csv",
    index=False
)


# ==========================================
# E. ASSOCIATION RULE MINING
# ==========================================

print("\nE. PROSES ASSOCIATION RULE MINING")

MIN_CONFIDENCE4 = 0.50

rules4 = association_rules(
    freq4,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE4
)

rules4 = rules4.sort_values(
    by="lift",
    ascending=False
).reset_index(drop=True)

kolom_rules4 = [
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction"
]

print("\nHASIL ASSOCIATION RULES:")

if len(rules4) > 0:
    print(
        rules4[kolom_rules4]
        .round(4)
        .to_string(index=False)
    )
else:
    print("Tidak ditemukan aturan asosiasi.")

print("\nJumlah Association Rules:", len(rules4))


# Menyimpan rules ke CSV
rules4_csv = rules4.copy()

rules4_csv["antecedents"] = rules4_csv["antecedents"].apply(
    lambda x: ", ".join(sorted(x))
)

rules4_csv["consequents"] = rules4_csv["consequents"].apply(
    lambda x: ", ".join(sorted(x))
)

rules4_csv.to_csv(
    "output/rules_toko_ruxxy.csv",
    index=False
)



# ==========================================
# F. VISUALISASI GRAFIK ASSOCIATION
# ==========================================

if len(rules4) > 0:

    plt.figure(figsize=(10, 6))

    sc = plt.scatter(
        rules4["support"],
        rules4["confidence"],
        s=100,
        c=rules4["lift"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="black"
    )

    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Grafik Association Rules - Toko Ruxxy")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "output/grafik_association.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("grafik_association.png berhasil dibuat!")

else:
    print("Tidak ada rules untuk divisualisasikan.")


# ==========================================
# G. VISUALISASI GRAFIK LIFT
# ==========================================

if len(rules4) > 0:

    top10 = rules4.head(10).copy()

    top10["aturan"] = top10.apply(
        lambda x:
        ", ".join(sorted(x["antecedents"]))
        + " -> " +
        ", ".join(sorted(x["consequents"])),
        axis=1
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        top10["aturan"],
        top10["lift"],
        edgecolor="black"
    )

    plt.xlabel("Nilai Lift")
    plt.ylabel("Aturan Asosiasi")
    plt.title("10 Association Rules dengan Lift Tertinggi")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        "output/grafik_lift.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("grafik_lift.png berhasil dibuat!")

else:
    print("Tidak ada rules untuk divisualisasikan.")


# ==========================================
# H. REKOMENDASI BISNIS
# ==========================================

print("\nH. REKOMENDASI BISNIS TOKO RUXXY")

if len(rules4) > 0:

    top3 = rules4.head(3)

    for i, (_, row) in enumerate(top3.iterrows(), 1):

        antecedent = ", ".join(sorted(row["antecedents"]))
        consequent = ", ".join(sorted(row["consequents"]))

        print(f"\nREKOMENDASI {i}")
        print("Jika pelanggan membeli:", antecedent)
        print("Maka produk yang direkomendasikan:", consequent)

        print(f"Support    : {row['support']:.4f}")
        print(f"Confidence : {row['confidence']:.4f}")
        print(f"Lift       : {row['lift']:.4f}")

        print(
            "Strategi   : Toko dapat menerapkan "
            "bundling atau cross-selling berdasarkan "
            "hubungan kedua produk tersebut."
        )

else:
    print("Belum ada aturan yang dapat dijadikan rekomendasi.")


# ==========================================
# I. KESIMPULAN
# ==========================================

print("\n" + "=" * 60)
print("KESIMPULAN TUGAS 4")
print("=" * 60)

print(
    "Association Rule Mining berhasil diterapkan "
    "pada dataset transaksi Toko Ruxxy."
)

print(
    "Analisis dilakukan menggunakan algoritma Apriori "
    "dengan minimum support 0.10 dan minimum confidence 0.50."
)

print(
    "Hasil analisis dapat digunakan untuk mengetahui "
    "hubungan pembelian antarproduk serta membantu "
    "menentukan strategi pemasaran toko."
)


# ==========================================
# J. DAFTAR FILE OUTPUT
# ==========================================

print("\nFILE OUTPUT BERHASIL DIBUAT:")

print("- dataset/transactions_toko_ruxxy.csv")
print("- dataset/transactions_toko_ruxxy_onehot.csv")
print("- output/frequent_itemsets_toko_ruxxy.csv")
print("- output/rules_toko_ruxxy.csv")
print("- output/grafik_lift.png")
print("- output/grafik_association.png")

print("\nTUGAS 4 SELESAI!")
