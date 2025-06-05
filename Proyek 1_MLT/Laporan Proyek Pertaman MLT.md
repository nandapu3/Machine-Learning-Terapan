# Laporan Proyek Machine Learning - Putri Nanda Sari

## Domain Proyek

Ketahanan pangan merupakan salah satu aspek penting dalam menjaga kesejahteraan dan stabilitas sebuah negara. Mewujudkan ketahanan pangan yang berkelanjutan telah menjadi topik penting dan prioritas utama dalam berbagai pertemuan yang diadakan oleh berbagai negara dan organisasi internasional [1]. Salah satu indikator utama ketahanan pangan adalah tercapainya produksi pangan yang konsisten dan mengalami peningkatan setiap tahunnya. Namun, hasil panen sangat dipengaruhi oleh berbagai faktor lingkungan, seperti curah hujan dan suhu rata-rata. Perubahan iklim dan penerapan input pertanian salah satunya pestisida, dapat menimbulkan ketidakpastian dalam produksi pangan [2]. Perubahan iklim dan penerapan input pertanian, termasuk pestisida, dapat menimbulkan ketidakpastian dalam produksi pangan, yang apabila terjadi gagal panen, dapat mengancam kestabilan ketahanan pangan nasional maupun global [3].
Kemajuan teknologi yang semakin pesat, terutama dalam bidang data dan machine learning, membuka peluang besar untuk penerapan analisis prediktif dalam memodelkan dan memproyeksikan hasil panen berdasarkan berbagai variabel yang memengaruhinya. Analisis prediktif ini sangat penting karena dapat memberikan gambaran awal mengenai potensi hasil panen yang akan diperoleh berdasarkan faktor-faktor tersebut. Oleh karena itu, penerapan metode prediksi berbasis teknologi, terutama machine learning, menjadi langkah strategis untuk mendukung pengambilan keputusan yang lebih efektif dalam mengurangi risiko gagal panen yang pada akhirnya berpotensi mengakibatkan kekurangan pangan [4]. Dengan adanya proyeksi awal ini, pihak terkait dapat melakukan evaluasi dan pengambilan keputusan yang lebih baik guna mencegah terjadinya kekurangan pangan.
Dalam membangun model analisis prediksi hasil panen ini, digunakan pendekatan regresi multivariat yang memungkinkan penggabungan berbagai faktor secara simultan untuk menghasilkan prediksi yang akurat serta dapat diandalkan. Model ini dibangun menggunakan data produksi pertanian dan faktor lingkungan dari berbagai negara dan jenis tanaman. Data historis tersebut akan dilatih untuk mengenali pola hubungan antara variabel lingkungan dan hasil panen. Hasil prediksi dari model ini diharapkan dapat membantu pengambilan kebijakan serta petani dalam merencanakan strategi produksi dan mitigasi risiko terkait ketahanan pangan, sehingga mendukung ketahanan pangan global secara berkelanjutan.
Referensi
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
- Menerapkan beberapa algoritma machine learning regresi, seperti Random Forest Regressor dan Gradient Boosting Machines (XGBoost), untuk memodelkan hubungan antar variabel serta melakukan perbandingan kinerja model berdasarkan matrik evaluasi untuk memilih model yang paling akurat dan efektif
- Melakukan evaluasi model menggunakan metrik kuantitatif seperti MAE (Mean Absolute Error), RMSE, dan R-squared untuk mengukur akurasi prediksi hasil panen oleh masing-masing model.


## Data Understanding
Dataset yang digunakan dalam proyek ini bersumber dari data open source Crop Yield Prediction Dataset di Kaggle yang berisikan informasi berupa data produksi pertanian dan faktor lingkungan dari berbagai negara dan jenis tanaman selama rentang tahun 1990-2013. Dataset ini terdiri atas 28.242 entri dengan 7 kolom yang mencakup informasi penting, memungkinkan analisis dan prediksi hasil panen dengan mempertimbangkan berbagai variabel iklim dan agrikultural yang berkontribusi terhadap ketahanan pangan global.  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

