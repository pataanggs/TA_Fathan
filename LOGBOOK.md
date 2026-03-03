# Research Logbook

## June 2025

|     DATE     | TASK                                                                                                           |
| :----------: | -------------------------------------------------------------------------------------------------------------- |
| 30 JUNE 2025 | Mengambil fokus literature dan mulai mencari beberapa paper yang relevan dengan penelitian yang akan dilakukan |

## July 2025

|       DATE        | TASK                                                                                                                                                                                                    |
| :---------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1- 3 JULY 2025   | Mereview beberapa paper yang sudah ditemukan menggunakan bantuan AI [NotebookLM](https://notebooklm.google.com/notebook/366e72a8-a3fe-4574-a245-581382484aeb)                                           |
| 24 - 31 JULY 2025 | Mencoba mengulik how to use Whisper untuk transkrip audio menjadi teks, comparing semua model yang tersedia (small, medium, large, turbo) menggunakan audio files yang diambil dari Youtube dan Podcast |

## August 2025

|      DATE      | TASK                                                                                                  |
| :------------: | ----------------------------------------------------------------------------------------------------- |
| 15 AUGUST 2025 | Mencari dataset audio berbahasa indonesia ataupun berbahasa daerah yang ada di Indonesia              |
| 29 AUGUST 2025 | Masih mencari dataset audio berbahasa indonesia atau bahasa daerah indonesia (Sunda, jawa, Bali, dll) |

## September 2025

|          DATE          | TASK                                                                                                          |
| :--------------------: | ------------------------------------------------------------------------------------------------------------- |
| 10 - 12 September 2025 | Mencoba membuat workplan TA untuk target Sempro di bulan November                                             |
| 12 - 14 September 2025 | Mencoba breakdown dataset yang didapatkan untuk training model di colab. Gagal karena runtime exceed (24 jam) |
| 21 - 28 September 2025 | Mencoba menggunakan runpod dengan dataset yang sudah didapatkan. Masih gagal karena data collator error       |

## Oktober 2025

|      DATE       | TASK                                              |
| :-------------: | ------------------------------------------------- |
| 1 Oktober 2025  | Mencoba melakukan train                           |
| 7 Oktober 2025  | Mulai menuliskan BAB 1. Latar Belakang Penelitian |
| 24 Oktober 2025 | Revisi Bab 1                                      |

# November 2025

|       DATE       | TASK                                                                                        |
| :--------------: | ------------------------------------------------------------------------------------------- |
| 3 November 2025  | Mulai menulis BAB 2. Tinjauan Pustaka                                                       |
| 11 November 2025 | BAB 2 Selesai namun masih sangat kasar, butuh konsultasi terkait artikel pembanding         |
| 11 November 2025 | Mencoba menghubungi pihak Ikatan Keluarga Minang ITERA untuk meminta tolong membuat dataset |
| 17 November 2025 | Dataset yang terkumpul baru 2, Target adalah 50                                             |
| 27 November 2025 | Menulis BAB 3. Metode Penelitian                                                            |
| 27 November 2025 | BAB 3 selesai namun masih sangat kasar, butuh konsultasi terkait metode penelitian          |

# Desember 2025
|       DATE       | TASK                                                                                        |
| :--------------: | ------------------------------------------------------------------------------------------- |
| 12 Desember 2025 | Start training Freeze Encoder                                                               |
| 20 Desember 2025 | Mencoba training kembali dengan beberapa perubahan config untuk freeze encoder              |

# Januari 2026
|       DATE       | TASK                                                                                        |
| :--------------: | ------------------------------------------------------------------------------------------- |
| 10 Januari 2026  | Masih mencoba mengulik konfigurasi training yang pas untuk freeze encoder                   |
| 15 Januari 2026  | Research ulang penelitian terdahulu terkait konfigurasi yang sesuai                         |
| 22 Januari 2026  | Berhasil melakukan training pertama kali dengan WER dan CER yang cukup                      |
| 28 Januari 2026  | Mencoba melakukan training ulang dengan konfigurasi parameter yang berbeda                  |

# Februari 2026
|       DATE       | TASK                                                                                        |
| :--------------: | ------------------------------------------------------------------------------------------- |
| 1 Februari 2026  | Training ulang untuk mencari performa WER dan CER terbaik                                   |
| 4 Februari 2026  | Training terakhir freeze encoder                                                            |
| 8 Februari 2026  | Membuat config untuk training menggunakan PEFT LORA dan melakukan training LORA             |

# Maret 2026
|       DATE       | TASK                                                                                        |
| :--------------: | ------------------------------------------------------------------------------------------- |
| 2 Maret 2026     | Melakukan eksperimen Weighted Cross entropy                                                 |
| 3 Maret 2026     | Menuliskan BAB 4 dan BAB 5 dan merangkum hasil 18x run untuk kedua metode                   |