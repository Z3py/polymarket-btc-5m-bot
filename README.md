# Polymarket BTC Up/Down 5-Minute Bot

Bot Python untuk memantau market BTC Up/Down 5 menit di Polymarket, menghitung probabilitas `UP`/`DOWN`, dan hanya entry saat ada expected value positif, edge cukup, feed sehat, spread/slippage/liquidity valid, serta risk manager mengizinkan.

Bot ini **tidak menjanjikan winrate 80%, 90%, atau 95%**. Entry live memakai hard gate edge, EV, spread, liquidity, dan confidence. Performa aktual harus dievaluasi dari log, Brier score, calibration error, dan PnL.

## Safety Defaults

- Default selalu `PAPER_TRADING`.
- Real order hanya aktif jika `REAL_TRADING=true` dan credential lengkap.
- Semua order real memakai limit order.
- All-in diblokir. Maksimum posisi per trade selalu 10% modal.
- Jika sinyal tidak valid, bot `SKIP`.
- Tidak ada martingale, revenge trading, atau overtrading.
- Jika rolling winrate aktual 50 trade di bawah 60%, bot masuk `SAFE_MODE`.

## Struktur

```text
polymarket_btc_5m_bot/
|-- README.md
|-- requirements.txt
|-- .env.example
|-- main.py
|-- config.py
|-- polymarket_client.py
|-- market_resolver.py
|-- btc_data_feed.py
|-- feature_engineering.py
|-- models.py
|-- risk_manager.py
|-- execution_engine.py
|-- backtester.py
|-- database.py
|-- logger.py
|-- dashboard.py
`-- tests/
```

## Install

```bash
cd polymarket_btc_5m_bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`. Untuk paper mode, credential Polymarket boleh kosong:

```env
REAL_TRADING=false
PAPER_STARTING_BALANCE=1000
MARKET_URL=https://polymarket.com/id/event/btc-updown-5m-1778690700
```

Untuk real trading, isi credential resmi CLOB. Jangan hardcode key di kode.

## Jalankan Bot

```bash
python main.py --hours 24
```

Bot akan:

1. Resolve market dari `MARKET_URL` atau mencari BTC 5m aktif jika URL sudah expired.
2. Ambil orderbook UP/DOWN dari Polymarket CLOB.
3. Ambil BTC dari Binance dan Coinbase, plus Chainlink jika `CHAINLINK_RPC_URL` dan `CHAINLINK_BTC_USD_FEED_ADDRESS` tersedia.
4. Skip jika feed BTC beda terlalu jauh.
5. Generate fitur return, volatility, EMA microtrend, breakout, z-score, implied probability, dan orderbook imbalance.
6. Prediksi dengan ensemble logistic online, Bayesian adjustment, Kalman trend, rule microstructure, dan XGBoost opsional.
7. Hitung EV dan edge.
8. Risk manager menentukan size atau SKIP.
9. Simpan prediksi, trade, dan hasil ke SQLite.

## Uji 24 Jam Sebelum Real Trading

Untuk menguji edge dan winrate selama 24 jam, gunakan forward-test paper-only:

```bash
python backtester.py forward --hours 24
```

Command ini memaksa `REAL_TRADING=false`, meskipun `.env` disetel real. Setelah selesai, tampilkan laporan:

```bash
python backtester.py report --hours 24
```

Laporan menampilkan:

- total prediksi dan jumlah SKIP
- settled paper trades
- winrate
- average edge
- average EV
- total paper PnL
- profit factor
- max drawdown
- Brier score
- apakah SAFE_MODE diperlukan

Backtest historis murni membutuhkan data historis Polymarket orderbook/market price UP-DOWN. Tanpa data itu, edge terhadap harga market historis tidak bisa diuji secara jujur. Forward-test 24 jam adalah mode validasi awal yang paling aman sebelum real trading.

## Dashboard

```bash
streamlit run dashboard.py
```

Dashboard memakai output scorecard seperti contoh: hero metrics, verdict vs threshold, tab `Signals`, tab `Hot >=16%`, model/features snapshot, skip reasons, rolling winrate, Brier score, edge, confidence, trade log, dan status `SAFE_MODE`.

## Formula EV

```text
EV_UP = p_up * (1 - price_up) - (1 - p_up) * price_up - fees - slippage
EDGE_UP_ABS = p_up - price_up

EV_DOWN = p_down * (1 - price_down) - (1 - p_down) * price_down - fees - slippage
EDGE_DOWN_ABS = p_down - price_down
```

Entry hanya dipertimbangkan jika `EDGE >= 0.16`, EV positif, spread tidak lebih dari 4%, slippage tidak lebih dari 2%, liquidity minimal 3x order size, feed sehat, confidence minimal high threshold, dan time-to-expiry valid.

## Real Trading Guard Rails

`REAL_TRADING=true` tetap tidak cukup jika credential kurang. Bot akan gagal start. Credential yang dipakai:

```env
POLYMARKET_PRIVATE_KEY=
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_API_PASSPHRASE=
POLYMARKET_FUNDER=
REAL_TRADING=true
```

Jika rule all-in diminta, risk manager mengembalikan:

```text
ALL-IN REQUEST BLOCKED BY RISK MANAGER. Position capped at 10%.
```

## Tests

```bash
pytest
```

Test saat ini mencakup kalkulasi EV/edge dan guard rail risk manager: cap 10%, blokir all-in, skip saat spread lebar, dan pause setelah 3 loss.

## Catatan Operasional

- Polymarket CLOB read endpoints publik; order placement perlu SDK resmi dan auth.
- WebSocket market channel dipakai untuk update orderbook, dengan fallback polling REST.
- Feed Binance/Coinbase bisa terkena rate limit atau blokir regional. Jika salah satu gagal dan hanya satu sumber sehat, bot akan SKIP.
- Chainlink bersifat opsional karena butuh RPC dan alamat aggregator yang benar untuk network yang dipilih.
- Ini software trading berisiko tinggi. Mulai dari paper mode dan evaluasi kalibrasi sebelum real trading.
