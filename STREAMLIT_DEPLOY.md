# Deploy ke Streamlit Community Cloud

Error "The app's code is not connected to a remote GitHub repository" berarti Streamlit Cloud tidak bisa deploy dari app lokal `127.0.0.1`. Kode harus dipush ke GitHub dulu.

## Opsi Paling Mudah

1. Buat repo baru di GitHub, misalnya `polymarket-btc-5m-bot`.
2. Upload semua file di folder ini ke repo tersebut.
3. Pastikan file berikut ada di root repo:
   - `streamlit_app.py`
   - `dashboard.py`
   - `requirements.txt`
   - `config.py`
   - `database.py`
4. Buka Streamlit Community Cloud.
5. Klik `Create app`.
6. Pilih repository GitHub.
7. Pilih branch, biasanya `main`.
8. Main file path isi:

```text
streamlit_app.py
```

9. Deploy.

## Secrets / Environment

Untuk dashboard paper/backtest, secret tidak wajib. Jangan upload `.env` ke GitHub.

Jika nanti perlu environment variable di Streamlit Cloud, masukkan melalui `Advanced settings > Secrets`, bukan lewat file `.env`.

Minimal paper settings:

```toml
REAL_TRADING = "false"
PAPER_STARTING_BALANCE = "1000"
EDGE_HIGH = "0.16"
MIN_CONFIDENCE_HIGH = "85"
DATABASE_URL = "sqlite:///bot.sqlite3"
```

Catatan: SQLite di Streamlit Cloud tidak cocok untuk data trading live jangka panjang karena storage cloud bisa reset. Untuk dashboard public, lebih baik upload hasil backtest statis atau pakai database eksternal.

## Kenapa Tidak Bisa dari Localhost?

Streamlit Community Cloud deploy dari GitHub repository, bukan dari komputer lokal. Localhost hanya untuk preview lokal.
