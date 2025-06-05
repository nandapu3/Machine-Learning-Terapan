# Laporan Proyek Machine Learning - Putri Nanda Sari

## Domain Proyek

Ketahanan pangan merupakan salah satu aspek penting dalam menjaga kesejahteraan dan stabilitas sebuah negara. Mewujudkan ketahanan pangan yang berkelanjutan telah menjadi topik penting dan prioritas utama dalam berbagai pertemuan yang diadakan oleh berbagai negara dan organisasi internasional [1]. Salah satu indikator utama ketahanan pangan adalah tercapainya produksi pangan yang konsisten dan mengalami peningkatan setiap tahunnya. Namun, hasil panen sangat dipengaruhi oleh berbagai faktor lingkungan, seperti curah hujan dan suhu rata-rata. Perubahan iklim dan penerapan input pertanian salah satunya pestisida, dapat menimbulkan ketidakpastian dalam produksi pangan [2]. Perubahan iklim dan penerapan input pertanian, termasuk pestisida, dapat menimbulkan ketidakpastian dalam produksi pangan, yang apabila terjadi gagal panen, dapat mengancam kestabilan ketahanan pangan nasional maupun global [3].  
Kemajuan teknologi yang semakin pesat, terutama dalam bidang data dan machine learning, membuka peluang besar untuk penerapan analisis prediktif dalam memodelkan dan memproyeksikan hasil panen berdasarkan berbagai variabel yang memengaruhinya. Analisis prediktif ini sangat penting karena dapat memberikan gambaran awal mengenai potensi hasil panen yang akan diperoleh berdasarkan faktor-faktor tersebut. Oleh karena itu, penerapan metode prediksi berbasis teknologi, terutama machine learning, menjadi langkah strategis untuk mendukung pengambilan keputusan yang lebih efektif dalam mengurangi risiko gagal panen yang pada akhirnya berpotensi mengakibatkan kekurangan pangan [4]. Dengan adanya proyeksi awal ini, pihak terkait dapat melakukan evaluasi dan pengambilan keputusan yang lebih baik guna mencegah terjadinya kekurangan pangan.  
Dalam membangun model analisis prediksi hasil panen ini, digunakan pendekatan regresi multivariat yang memungkinkan penggabungan berbagai faktor secara simultan untuk menghasilkan prediksi yang akurat serta dapat diandalkan. Model ini dibangun menggunakan data produksi pertanian dan faktor lingkungan dari berbagai negara dan jenis tanaman. Data historis tersebut akan dilatih untuk mengenali pola hubungan antara variabel lingkungan dan hasil panen. Hasil prediksi dari model ini diharapkan dapat membantu pengambilan kebijakan serta petani dalam merencanakan strategi produksi dan mitigasi risiko terkait ketahanan pangan, sehingga mendukung ketahanan pangan global secara berkelanjutan.
### Referensi
[1]	A. Suryana, “Menuju Ketahanan Pangan Indonesia Berkelanjutan 2025: Tantangan dan Penanganannya,” forum penelit. agro ekon., vol. 32, no. 2, p. 123, Oct. 2014, doi: 10.21082/fae.v32n2.2014.123-135.  
[2]	G. Pérez-Lucas, G. Navarro, and S. Navarro, “Adapting agriculture and pesticide use in Mediterranean regions under climate change scenarios: A comprehensive review,” European Journal of Agronomy, vol. 161, p. 127337, Nov. 2024, doi: 10.1016/j.eja.2024.127337.  
[3]	M. Amin, L. Budiman, and D. Suhendi, “RESILIENSI PENGUATAN KETAHANAN PANGAN DAERAH DI INDONESIA,” Jurnal Perlindungan Masyarakat Bestuur Praesidium, vol. 1, no. 2, pp. 63–71, 2024.  
[4]	T. Nizami, M. A. Mustaqiim, and W. Ariannor, “Analisis Kinerja Model Machine learning dalam Prediksi Gagal Panen Gabah,” vol. 21, no. 1, pp. 184–192, 2025, doi: http://dx.doi.org/10.35889/progresif.v21i1.2501.  

## Business Understanding

### Problem Statements
Pernyataan masalah latar belakang:
- Hasil panen dipengaruhi signifikan oleh faktor lingkungan seprti perubahan iklim, curah hujan, hingga suhu, yang menimbulkan adanya ketidakpastian dalam produksi pangan.
- Ketidakpastian hasil panen dapat berujung pada kegagalan panen, yang kemudia berpotensi mengganggu stabilitas ketahanan pangan baik secara nasional maupun global.
- Saat ini, masih terbatas alat prediksi hasil panen yang cukup akurat dan mudah diakses, sehingga menyulitkan pengambilan keputusan yang efektif dan pelaksanaan mitigasi risiko secara optimal.
### Goals
Tujuan dari pernyataan masalah:
- Mengembangkan model prediksi hasil panen yang akurat dengan memanfaatkan faktor-faktor lingkungan utama.
- Menyediakan informasi proyeksi hasil panen yang dapat digunakan oleh pengambil kebijakan dan petani untuk perencanaan produksi serta mitigasi risiko gagal panen
- Mendukung upaya peningkatan ketahanan pangan nasional dan global melalui penerapan teknologi prediktif yang andal
### Solution statements
Solusi dari pernyataan masalah:
- Menerapkan beberapa algoritma machine learning regresi, seperti Linier Regression, Random Forest Regressor dan Gradient Boosting Machines (XGBoost), untuk memodelkan hubungan antar variabel serta melakukan perbandingan kinerja model berdasarkan matrik evaluasi untuk memilih model yang paling akurat dan efektif
- Melakukan evaluasi model menggunakan metrik kuantitatif seperti MAE (Mean Absolute Error), RMSE, dan R-squared untuk mengukur akurasi prediksi hasil panen oleh masing-masing model.


## Data Understanding
Dataset yang digunakan dalam proyek ini bersumber dari data open source [Crop Yield Prediction Dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fmrigaankjaswal%2Fcrop-yield-prediction-dataset) di Kaggle yang berisikan informasi berupa data produksi pertanian dan faktor lingkungan dari berbagai negara dan jenis tanaman selama rentang tahun 1990-2013. Dataset ini terdiri atas 28.242 entri dengan 7 kolom yang mencakup informasi penting, memungkinkan analisis dan prediksi hasil panen dengan mempertimbangkan berbagai variabel iklim dan agrikultural yang berkontribusi terhadap ketahanan pangan global.  

### Variabel-variabel pada Cro Yield Dataset adalah sebagai berikut:
-	**Area (Country)** : Kolom ini berisikan nama negara atau wilayah dimana data berasal, yang  merupakan kategorikal tanpa adanya missing value.
-	**Item** : Kolom berisikan jenis produk pertanian atau tanaman yang dianalisis, data ini berbentuk kategorikal.
-	**Year** : Tahun pengamatan data, yang merekam hasil panen dan variabel terkait secara longitudinal
-	**hg/ha_yield** : Variabel target berupa hasil panen hektogram per hektar (hg/ha), yang akan diprediksi.
-	**average_rain_fall_mm_per_year** : Rata-rata curah hujan tahunan (mm), faktor lingkungan utama yang memengaruhi produksi.
-	**pesticides_tonnes** : Jumlah pestisida yang digunakan dalam satuan ton, yang dapat berdampak pada hasil panen dan lingkungan.
-	**avg_temp** : Suhu rata-rata tahunan dalam derajat Celsius, yang mempengaruhi pertumbuhan dan hasil tanaman.

### Exploratory Data Analysis (EDA):
EDA dilakukan dengan memeriksa struktur data pada dataset yang digunakan. Dataset terdiri dari 28.242 entri dan 7 fitur. Pemeriksaan awal menggunakan `df.info()` menunjukkan tidak ada nilai yang hilang. `df.describe()` memberikan gambaran tentang sebaran data, menunjukkan nilai rata-rata dan standar deviasi yang konsisten untuk setiap fitur. Tidak ditemukan duplikasi setelah pengecekan dengan `df.duplicated()`. Untuk lebih memahami data dilakukan visualisasi untuk beberapa poin berikut. 
-	**Matriks Boxplot**
  ![Boxplot](https://github.com/nandapu3/Machine-Learning-Terapan/blob/main/Proyek%201_MLT/download.png)
Pada gambar diagram boxplot memperlihatkan bahwa fitur `hg/ha_yield` dan `pesticides_tonnes` memiliki banyak pencilan (ditunjukkan oleh titik-titik di luar whisker), menunjukkan adanya nilai ekstrim yang mungkin perlu diperhatikan atau ditangani lebih lanjut.
-	**Matriks Distribusi**
  ![Histogram](https://github.com/nandapu3/Machine-Learning-Terapan/blob/main/Proyek%201_MLT/download%20(1).png)
Histogram Distribusi memperlihatkan bentuk distribusi dari setiap fitur. `hg/ha_yield` dan `pesticides_tonnes` memiliki distribusi miring ke kanan (right-skewed), menunjukkan banyak nilai rendah dan beberapa nilai yang sangat tinggi.

## Data Preparation
Data preparation dilakukan dengan menerapkan beberapa teknik yang diperlukan berdasarkan hasil yang diperoleh dari data understanding sebelumnya. Langkah-langkah Data Preparation ini penting untuk memastikan data dalam kondisi terbaik untuk pemodelan. Berikut ini beberapa teknik data preparation yang diterapkan:
- **Menghapus Kolom yang tidak diperlukan:** Kolom atau fitur yang tidak memberikan informasi relevan (seperti: `Unnamed: 0`) dihapus dari dataset. Menghapus kolom yang tidak penting dilakukan untuk mencegah adanya noise dalam data dan memastikan model hanya bekerja dengan fitur yang relevan.
- **Transformasi Logaritmik:** Transformasi logaritmik diterapkan pada dua fitur yang diketahui memiliki distribusi miring. Fitur ini mencakup `hg/ha_yield` dan `pesticides_tonnes` yang juga sebelumnya diketahui memiliki outlier. Namun, datanya bukan merupakan kesalahan input dan tidak bisa semata-mata dihapus. Transformasi logaritmik diterapkan dengan menggunakan `np.log1p()`. Data yang tidak terdistribusi normal dapat mempengaruhi kinerja model. **Log-transformation** membantu mendekatkan distribusi data ke distribusi normal dan meningkatkan kestabilan model.
- **Encoding Kategorikal:** Mengonversi fitur kategorikal yaitu fitur `Area` dan `Item` menjadi angka menggunakan **Label Encoding**. Model regresi tidak bisa langsung bekerja dengan data kategorikal. **Label Encoding** mengubah kategori menjadi format numerik yang dapat diproses oleh model.
- **Feature Scaling:** Menstandarisasi fitur numerik menggunakan **StandardScaler** untuk memastikan semua fitur berada dalam rentang yang seragam. Beberapa algoritma cukup sensitif terhadap skala fitur yang berbeda, sehingga **feature scaling** diterapkan untuk memastikan bahwa fitur dengan skala besar tidak memdominasi model.
- **Data Splitting:** Membagi dataset menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`. Pembagian dataset ini bertujuan agar model dapat dilatih dengan data latih yang cukup banyak dan diuji dengan data uji untuk mengevaluasi kinerjanya.

## Modeling
Dalam pemodelan yang dilakukan dengan pendekatan regresi multivariat, beberapa model diterapkan pada data pelatihan untuk melihat kemungkinan model yang memiliki performa baik pada data yang dimiliki. Berikut ini beberapa model yang digunakan:

- **Model Linear Regression**  
  Model ini digunakan untuk memodelkan hubungan linear antara fitur dan target. Proses pemodelan dengan linear regression menemukan garis terbaik yang meminimalkan kesalahan antara nilai prediksi dan aktual. Pada pemodelan yang dilakukan, tidak ada parameter khusus yang diatur dan hanya menggunakan default setting.  
  **Kelebihan:**  Model ini sederhana, mudah, dan cepat. Model ini juga mudah untuk diintegrasikan.  
  **Kekurangan:**  Model ini tidak bisa menangani hubungan non-linear, juga sangat sensitif terhadap outliers.

- **Model Random Forest Regression**  
  Random Forest menggabungkan banyak pohon keputusan untuk menghasilkan prediksi yang lebih stabil dan akurat. Model ini dilatih dengan subset data acak dan memilih fitur acak pada setiap pohon. Parameter yang digunakan selama proses pemodelan yaitu `n_estimators` untuk jumlah pohon, `random_state` untuk konsistensi hasil.  
  **Kelebihan:**  Model ini dapat menangani hubungan non-linear dan interaksi antar fitur, serta lebih robust terhadap overfitting.  
  **Kekurangan:**  Memerlukan sumber komputasi dan waktu pelatihan yang lebih banyak karena model tergolong kompleks.

- **Model XGBoost Regression**  
  XGBoost menggunakan teknik gradient boosting untuk membangun pohon keputusan bertahap yang saling memperbaiki kesalahan. Parameter yang digunakan dalam proses pemodelan adalah `objective='reg:squarederror'` untuk regresi, `n_estimators` untuk jumlah pohon, dan `random_state` untuk konsistensi.  
  **Kelebihan:**  Sangat efektif untuk dataset besar dan kompleks, memberikan akurasi yang sangat tinggi.  
  **Kekurangan:**  Membutuhkan tuning hyperparameter lebih intensif dan lebih lambat dalam pelatihan dibandingkan Random Forest.

## Evaluation
Pada tahap evaluasi model regresi ini, beberapa metrik digunakan untuk menilai performa model yang dibangun, yaitu **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, dan **R-squared (R²)**.

- **Mean Absolute Error (MAE):** mengukur rata-rata kesalahan absolut antara nilai yang diprediksi dan nilai yang sebenarnya. Semakin rendah MAE, semakin baik model memprediksi target.
- **Root Mean Squared Error (RMSE):** mengukur akar dari rata-rata kesalahan antara nilai yang diprediksi dan nilai sebenarnya. RMSE menambahkan bobot pada kesalahan besar, sehingga lebih sensitif terhadap outliers.
- **R-Squared (R²):** metrik ini mengukur seberapa baik sebuah model dalam menjelaskan variansi dalam data. Nilai R² berada dalam kisaran rentang 0 dan 1, dimana 1 menunjukkan model yang sempurna.

Berdasarkan hasil evaluasi yang dilakukan menggunakan tiga metrik yang telah disebutkan, proyek ini memperoleh kinerja masing-masing dari tiga model yang dilatih, sebagai berikut:

- **Linear Regression**  
  Model Linear Regression menunjukkan kinerja yang kurang baik dengan R-squared sebesar 0.1075. Ini berarti bahwa model hanya mampu menjelaskan sekitar 10.75% dari variansi dalam data, yang menunjukkan bahwa hubungan linier antara fitur dan hasil panen tidak cukup kuat. MAE sebesar 0.7965 dan RMSE sebesar 0.8891 mengindikasikan bahwa prediksi yang dihasilkan oleh model ini memiliki kesalahan yang cukup besar. Dengan hasil evaluasi tersebut, dapat disimpulkan bahwa Linear Regression tidak efektif untuk memprediksi hasil panen pada dataset ini, karena model ini tidak dapat menangani hubungan yang lebih kompleks atau non-linier dalam data.

- **Random Forest Regression**  
  Random Forest Regression menunjukkan hasil yang sangat baik dengan R-squared mencapai 0.9825, yang berarti model ini mampu menjelaskan 98.25% dari variansi dalam data. Dengan MAE yang sangat rendah (0.0605) dan RMSE (0.0175), model ini memberikan prediksi yang sangat akurat dan stabil. Random Forest mampu menangani hubungan non-linier antara fitur dan target serta mengurangi overfitting dengan baik. Hasil ini menunjukkan bahwa Random Forest adalah model terbaik di antara yang diuji, karena memberikan hasil yang sangat baik dalam memprediksi hasil panen.

- **XGBoost Regression**  
  XGBoost Regression juga memberikan hasil yang sangat baik dengan R-squared sebesar 0.9681, yang menunjukkan bahwa model ini mampu menjelaskan 96.81% dari variansi dalam data. Dengan MAE sebesar 0.1164 dan RMSE 0.0318, model ini memberikan akurasi yang tinggi meskipun sedikit lebih rendah dibandingkan dengan Random Forest. XGBoost dapat menangani dataset besar dan kompleks dengan baik, namun membutuhkan lebih banyak waktu pelatihan dan tuning hyperparameter untuk mendapatkan hasil optimal. Meskipun demikian, XGBoost tetap menjadi model yang sangat efektif dan memberikan hasil yang hampir setara dengan Random Forest.
