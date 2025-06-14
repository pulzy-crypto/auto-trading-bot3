import pandas as pd
import numpy as np
import ta
import ccxt
import os
import time
from datetime import datetime, time as dt_time, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SupertrendLiveBot:
    def __init__(self, 
                 symbol='BTC/USDT', # Contoh: Bitcoin/USDT
                 timeframe='3m',    # Timeframe 3 menit
                 atr_period=10,     # Periode ATR untuk Supertrend
                 factor=3.0,        # Faktor ATR untuk Supertrend
                 base_risk=0.8,     # Risiko per trade dalam USD
                 rr_asia=3.0,       # Risk-Reward Ratio untuk Asia (1:3)
                 rr_ln=10.0,        # Risk-Reward Ratio untuk LN/NY (Fixed TP 1:10)
                 trailing_rr=8.0,   # Trailing SL Ratio untuk LN/NY (1:8)
                 volume_factor=2.0, # Faktor untuk Volume Spike
                 fee_rate=0.0004,   # Tingkat biaya (0.04% untuk maker/taker)
                 api_key=None,
                 api_secret=None):

        self.symbol = symbol
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.factor = factor
        self.base_risk = base_risk
        self.rr_asia = rr_asia
        self.rr_ln = rr_ln
        self.trailing_rr = trailing_rr # Diperlukan untuk trailing SL
        self.volume_factor = volume_factor
        self.fee_rate = fee_rate

        # Presisi dan tick size akan di-overwrite dari exchange info
        self.price_decimals = None 
        self.max_qty_decimals = None
        self.tick_size = None 
        
        # Inisialisasi Binance exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key or os.getenv('BINANCE_API_KEY'),
            'secret': api_secret or os.getenv('BINANCE_SECRET_KEY'),
            'options': {
                'defaultType': 'future', # Untuk trading futures
                # 'testnet': True, # Uncomment ini jika Anda ingin menggunakan Binance Testnet
            },
            'enableRateLimit': True, # Mengatur batas request agar tidak dibanned
        })
        
        # Melacak status posisi
        self.position = {
            'status': 'NONE',    # 'NONE', 'OPEN'
            'type': None,        # 'LONG' or 'SHORT'
            'entry_price': None,
            'qty': 0,
            'sl': None,          # Stop Loss price (internal tracking)
            'tp': None,          # Take Profit price (internal tracking)
            'risk_amount': 0,    # Amount risked per trade
            'sl_order_id': None, # Binance SL order ID
            'tp_order_id': None, # Binance TP order ID
            'is_asia_entry': None, # True jika entry di sesi Asia, False jika LN/NY
            'entry_time': None   # Waktu entry posisi
        }

        # Coba load market info
        try:
            self.load_market_info()
        except Exception as e:
            print(f"ERROR: Gagal memuat informasi pasar untuk {self.symbol}: {e}")
            print("Pastikan API Key dan Secret Key sudah benar, dan symbol tersedia.")
            exit() # Keluar jika tidak bisa memuat info pasar

    def load_market_info(self):
        # Memuat info pasar untuk menentukan presisi harga dan kuantitas
        markets = self.exchange.load_markets()
        if self.symbol not in markets:
            raise Exception(f"Symbol {self.symbol} tidak ditemukan di Binance.")
        
        market = markets[self.symbol]
        
        # Ambil presisi dari market info
        # 'precision' memberikan jumlah desimal secara langsung
        self.price_decimals = market['precision']['price']
        self.max_qty_decimals = market['precision']['amount']
        
        # 'tickSize' adalah langkah harga minimum
        self.tick_size = market['limits']['price']['min'] 
        
        print(f"INFO: Informasi pasar untuk {self.symbol} dimuat.")
        print(f"  Price Precision (Decimal Places): {self.price_decimals}")
        print(f"  Quantity Precision (Decimal Places): {self.max_qty_decimals}")
        print(f"  Min Price Tick Size: {self.tick_size}")

    def _round_price(self, price):
        # Membulatkan harga sesuai kelipatan tick_size, bukan hanya desimal
        # Ini lebih robust karena tick_size adalah acuan minimum perubahan harga
        if self.tick_size is None:
            # Fallback jika tick_size belum dimuat (seharusnya tidak terjadi)
            return self.exchange.decimal_to_precision(
                price, 
                ccxt.ROUND, 
                self.price_decimals, 
                self.exchange.precision_mode
            )
        
        # Bulatkan harga ke kelipatan terdekat dari tick_size
        # Contoh: price=69000.01, tick_size=0.1 -> round(69000.01 / 0.1) * 0.1 = round(690000.1) * 0.1 = 690000 * 0.1 = 69000.0
        # Contoh: price=69000.05, tick_size=0.1 -> round(69000.05 / 0.1) * 0.1 = round(690000.5) * 0.1 = 690001 * 0.1 = 69000.1
        rounded_price = round(price / self.tick_size) * self.tick_size
        
        # Sekarang, pastikan jumlah desimal juga sesuai dengan price_decimals
        # Ini penting agar format string yang dikirim ke exchange sesuai harapan
        return self.exchange.decimal_to_precision(
            rounded_price,
            ccxt.ROUND, # Atau TRUNCATE tergantung kebutuhan, ROUND lebih umum
            self.price_decimals, # Gunakan price_decimals yang di-fetch dari exchange info
            self.exchange.precision_mode
        )

    def _round_qty(self, qty):
        # Membulatkan kuantitas sesuai presisi exchange
        return self.exchange.decimal_to_precision(
            qty, 
            ccxt.TRUNCATE, # Umumnya kuantitas dipotong, tidak dibulatkan ke atas
            self.max_qty_decimals, 
            ccxt.DECIMAL_TO_PRECISION # Mode padding default
        )

    def fetch_ohlcv(self, limit=200):
        # Mengambil data candlestick dari Binance
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"ERROR: Gagal mengambil data OHLCV: {e}")
            return pd.DataFrame()

    def calculate_supertrend(self, df):
        # Menggunakan fungsi supertrend dari library `ta`
        if len(df) < self.atr_period:
            print("WARNING: Data tidak cukup untuk menghitung Supertrend.")
            return df
        
        try:
            # Coba akses supertrend dari ta.trend (ini yang umum di versi baru)
            df['ST'], df['ST_Direction'] = ta.trend.supertrend(
                df['High'], df['Low'], df['Close'], 
                window=self.atr_period, 
                factor=self.factor, 
                fillna=True
            )
        except AttributeError:
            # Jika tidak ada di ta.trend, coba akses langsung dari ta (ini mungkin di versi yang sangat lama)
            print("WARNING: 'supertrend' not found in 'ta.trend', attempting to load from 'ta' directly.")
            try:
                df['ST'], df['ST_Direction'] = ta.supertrend(
                    df['High'], df['Low'], df['Close'], 
                    window=self.atr_period, 
                    factor=self.factor, 
                    fillna=True
                )
            except AttributeError:
                print("CRITICAL ERROR: 'supertrend' function not found in 'ta.trend' or 'ta'. Please check 'ta' library version.")
                return df # Mengembalikan df apa adanya, tapi ini akan menyebabkan error lain nantinya
        
        # ST_Direction: 1 for uptrend, -1 for downtrend
        # ST: Supertrend line itself

        return df

    def add_indicators(self, df):
        if len(df) < max(50, 200, 14, 10): # Minimal data untuk EMA200
            print("WARNING: Data tidak cukup untuk menghitung semua indikator.")
            return df
            
        df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50) # EMA25 di Pinescript, kita gunakan EMA50
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200) # EMA200 di Pinescript, kita gunakan EMA200
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Spike'] = df['Volume'] > (df['Volume_SMA'] * self.volume_factor)

        df['Bullish_Engulfing'] = (
            (df['Close'].shift(1) < df['Open'].shift(1)) & # Previous candle was bearish
            (df['Close'] > df['Open']) &                     # Current candle is bullish
            (df['Close'] > df['Open'].shift(1)) &            # Current close above previous open
            (df['Open'] < df['Close'].shift(1))              # Current open below previous close
        )

        df['Bearish_Engulfing'] = (
            (df['Close'].shift(1) > df['Open'].shift(1)) & # Previous candle was bullish
            (df['Close'] < df['Open']) &                     # Current candle is bearish
            (df['Close'] < df['Open'].shift(1)) &            # Current close below previous open
            (df['Open'] > df['Close'].shift(1))              # Current open above previous close
        )

        df['Body'] = abs(df['Close'] - df['Open'])
        df['Min_Body'] = df['ATR'] * 0.3 # Anda harus memastikan 'ATR' sudah ada di df
        df['Valid_Candle'] = df['Body'] >= df['Min_Body']
        return df

    def apply_time_filters(self, df):
        df['Hour'] = df.index.hour
        # Sesi Asia (UTC): 00:00 - 07:00 UTC
        # Sesi London/New York (UTC): 07:00 - 24:00 UTC
        df['Is_Asia'] = (df['Hour'] >= 0) & (df['Hour'] < 7)
        
        # Menentukan rasio SL Awal dan TP Tetap berdasarkan sesi
        df['RR_SL_Initial'] = df.apply(lambda x: self.rr_asia if x['Is_Asia'] else self.trailing_rr, axis=1)
        df['RR_TP_Fixed'] = df.apply(lambda x: self.rr_asia if x['Is_Asia'] else self.rr_ln, axis=1)
        
        return df.drop(columns=['Hour'])

    def calculate_position_sizes(self, entry_price, atr_value, rr_sl_initial):
        # Calculate risk amount per unit based on entry and initial stop loss
        # The 'rr_sl_initial' determines the initial distance of SL from entry (e.g., 3 * ATR or 8 * ATR)
        risk_per_unit = atr_value * rr_sl_initial

        # Ensure risk_per_unit is at least tick_size to avoid division by zero or too large qty
        if risk_per_unit < self.tick_size:
            risk_per_unit = self.tick_size
        
        # Qty dihitung berdasarkan base_risk dan risk_per_unit
        # Fee dianggap untuk round trip (beli dan jual), jadi 2x fee_rate
        qty = self.base_risk / (risk_per_unit + (entry_price * self.fee_rate * 2)) # Mempertimbangkan biaya di risk per unit

        return self._round_qty(qty)

    def check_signals(self, df):
        # Pastikan kita punya data terbaru yang lengkap
        if df.empty or len(df) < (self.atr_period + 200 + 2): # Minimal data untuk EMA200 + ATR + 2 bar
            return False, False, {}

        # Ambil baris terakhir yang sudah dihitung indikator
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] # Untuk Supertrend direction check

        # Pastikan semua kolom yang dibutuhkan ada dan bukan NaN
        required_cols = ['Close', 'ST', 'ST_Direction', 'Bullish_Engulfing', 
                         'Bearish_Engulfing', 'EMA50', 'EMA200', 'RSI', 'Valid_Candle', 
                         'Volume_Spike', 'ATR', 'RR_SL_Initial', 'RR_TP_Fixed']
        
        # Pastikan ATR juga tidak NaN pada candle terakhir
        if any(pd.isna(last_candle[col]) for col in required_cols) or pd.isna(last_candle['ATR']):
            print("DEBUG: Beberapa indikator atau data NaN di candle terakhir. Mengabaikan sinyal.")
            return False, False, {}

        # Logika Sinyal (sesuai Pinescript)
        # EMA2 (di Pinescript) diganti EMA50 (di Python), EMA25 (di Pinescript) diganti EMA200 (di Python)
        long_cond = (
            (last_candle['ST_Direction'] == 1) and # Supertrend saat ini uptrend
            (prev_candle['ST_Direction'] == -1) & # Supertrend baru saja berubah arah ke atas
            last_candle['Bullish_Engulfing'] and
            (last_candle['EMA50'] > last_candle['EMA200']) and # Corresponds to trend_up in Pinescript (ema2 > ema25)
            (last_candle['RSI'] > 50) and
            last_candle['Valid_Candle'] and
            last_candle['Volume_Spike']
        )

        short_cond = (
            (last_candle['ST_Direction'] == -1) and # Supertrend saat ini downtrend
            (prev_candle['ST_Direction'] == 1) &  # Supertrend baru saja berubah arah ke bawah
            last_candle['Bearish_Engulfing'] and
            (last_candle['EMA50'] < last_candle['EMA200']) and # Corresponds to trend_down in Pinescript (ema2 < ema25)
            (last_candle['RSI'] < 50) and
            last_candle['Valid_Candle'] and
            last_candle['Volume_Spike']
        )
        
        # Tambahan info untuk trade
        trade_info = {
            'entry_price': self._round_price(last_candle['Close']), # Harga entry dibulatkan
            'atr_value': last_candle['ATR'],
            'is_asia_entry': last_candle['Is_Asia'],
            'rr_sl_initial': last_candle['RR_SL_Initial'], # 3.0 for Asia, 8.0 for LN/NY
            'rr_tp_fixed': last_candle['RR_TP_Fixed']      # 3.0 for Asia, 10.0 for LN/NY
        }

        return long_cond, short_cond, trade_info

    def _cancel_order(self, order_id, symbol):
        """Membatalkan order spesifik di Binance."""
        try:
            if order_id:
                self.exchange.cancel_order(order_id, symbol)
                print(f"INFO: Order {order_id} dibatalkan.")
                return True
        except ccxt.OrderNotFound:
            print(f"WARNING: Order {order_id} tidak ditemukan atau sudah terisi/dibatalkan.")
            return False
        except Exception as e:
            print(f"ERROR: Gagal membatalkan order {order_id}: {e}")
            return False
        return False

    def _place_sl_order(self, position_type, qty, sl_price):
        """Menempatkan atau menempatkan ulang Stop Market SL order."""
        sl_order_id = None
        sl_side = 'SELL' if position_type == 'LONG' else 'BUY'
        
        try:
            sl_order = self.exchange.create_order(
                self.symbol,
                'STOP_MARKET', 
                sl_side, 
                qty, 
                None, # Price: None for market order
                {
                    'stopPrice': self._round_price(sl_price),
                    'timeInForce': 'GTC' # Good Till Cancelled
                }
            )
            sl_order_id = sl_order['id']
            print(f"INFO: SL STOP_MARKET order ditempatkan: Side={sl_side}, TriggerPrice={self._round_price(sl_price)} Qty={qty:.{self.max_qty_decimals}f}. Order ID: {sl_order_id}")
        except Exception as e:
            print(f"ERROR: Gagal menempatkan SL order: {e}")
            raise # Re-raise untuk ditangkap di manage_position atau run

        return sl_order_id

    def place_entry_order(self, trade_type, qty, entry_price):
        """Menempatkan LIMIT order untuk entry."""
        order = None
        side = 'BUY' if trade_type == 'LONG' else 'SELL'
        try:
            order = self.exchange.create_limit_order(self.symbol, side, qty, self._round_price(entry_price))
            print(f"INFO: LIMIT {side} order ditempatkan: Price={self._round_price(entry_price)} Quantity={qty:.{self.max_qty_decimals}f}. Order ID: {order['id']}")
            return order
        except ccxt.InsufficientFunds as e:
            print(f"ERROR: Dana tidak cukup untuk menempatkan order: {e}")
        except ccxt.InvalidOrder as e:
            print(f"ERROR: Order tidak valid (misal, harga terlalu jauh dari market): {e}")
        except Exception as e:
            print(f"ERROR: Gagal menempatkan entry LIMIT order: {e}")
        return None

    def place_tp_order(self, position_type, qty, tp_price):
        """Menempatkan LIMIT order untuk Take Profit."""
        tp_order_id = None
        tp_side = 'SELL' if position_type == 'LONG' else 'BUY'
        try:
            tp_order = self.exchange.create_limit_order(
                self.symbol, 
                tp_side, 
                qty, 
                self._round_price(tp_price), 
                {'reduceOnly': True, 'timeInForce': 'GTC'} # Penting untuk Binance Futures
            )
            tp_order_id = tp_order['id']
            print(f"INFO: TP LIMIT order ditempatkan: Side={tp_side}, Price={self._round_price(tp_price)} Qty={qty:.{self.max_qty_decimals}f}. Order ID: {tp_order_id}")
        except Exception as e:
            print(f"ERROR: Gagal menempatkan TP LIMIT order: {e}")
            raise # Re-raise untuk ditangkap di manage_position atau run
        return tp_order_id

    def manage_position(self, current_price, current_atr):
        if self.position['status'] == 'NONE':
            return # Tidak ada posisi terbuka

        position_type = self.position['type']
        entry_price = self.position['entry_price']
        qty = self.position['qty']
        
        # Cek apakah TP/SL sudah terisi oleh Binance
        try:
            if self.position['sl_order_id']:
                sl_status = self.exchange.fetch_order(self.position['sl_order_id'], self.symbol)
                if sl_status['status'] == 'closed' or sl_status['filled'] > 0:
                    print(f"INFO: SL order {self.position['sl_order_id']} terisi. Posisi ditutup via SL.")
                    self.close_position_after_fill('STOP_LOSS', current_price)
                    return
            if self.position['tp_order_id']:
                tp_status = self.exchange.fetch_order(self.position['tp_order_id'], self.symbol)
                if tp_status['status'] == 'closed' or tp_status['filled'] > 0:
                    print(f"INFO: TP order {self.position['tp_order_id']} terisi. Posisi ditutup via TP.")
                    self.close_position_after_fill('TAKE_PROFIT', current_price)
                    return
        except ccxt.OrderNotFound:
            print(f"WARNING: Order SL/TP {self.position.get('sl_order_id')} / {self.position.get('tp_order_id')} tidak ditemukan di Binance. Mungkin sudah terisi atau dibatalkan secara manual.")
            # Dalam kasus ini, cek posisi aktual di exchange
            try:
                balance = self.exchange.fetch_balance()
                current_positions = balance['info']['positions']
                symbol_position = [p for p in current_positions if p['symbol'] == self.symbol.replace('/USDT', 'USDT')][0]
                actual_qty = float(symbol_position['positionAmt'])

                if abs(actual_qty) < self._round_qty(0.001): # Periksa jika posisi aktual sudah sangat kecil atau nol
                    print(f"INFO: Posisi {self.symbol} sudah nol di Binance. Mereset state bot.")
                    self._reset_position_state()
                    return
                else:
                    print(f"WARNING: Posisi {self.symbol} masih ada di Binance ({actual_qty}) tapi order tidak terlacak. Perlu intervensi manual.")
                    # Jika ini terjadi, bot mungkin dalam keadaan tidak sinkron.
                    # Untuk keamanan, kita bisa mencoba menutup posisi.
                    # self.close_position(current_price, "MANUAL_CLOSE_DESYNC") 
                    # Atau biarkan dan log, tapi ini risiko.
            except Exception as ex:
                print(f"ERROR: Gagal memeriksa posisi aktual di Binance: {ex}")
        except Exception as e:
            print(f"ERROR: Gagal memeriksa status order di Binance: {e}")
        
        # Trailing Stop Loss Logic (Hanya untuk sesi LN/NY - jika entry_time BUKAN di sesi Asia)
        if not self.position['is_asia_entry'] and self.trailing_rr > 0:
            current_sl_internal = self.position['sl'] # SL yang sedang dilacak bot
            
            if position_type == 'LONG':
                # SL baru harus lebih tinggi dari SL sebelumnya
                # Hitung SL baru berdasarkan current_price dan trailing_rr
                new_sl_price = self._round_price(current_price - (current_atr * self.trailing_rr))
                if new_sl_price > current_sl_internal:
                    print(f"INFO: Mengupdate Trailing SL untuk LONG. Old SL: {current_sl_internal} -> New SL: {new_sl_price}")
                    self._cancel_order(self.position['sl_order_id'], self.symbol)
                    try:
                        new_sl_order_id = self._place_sl_order(position_type, qty, new_sl_price)
                        self.position['sl'] = new_sl_price
                        self.position['sl_order_id'] = new_sl_order_id
                    except Exception as e:
                        print(f"CRITICAL ERROR: Gagal menempatkan ulang SL order saat trailing: {e}. Menutup posisi untuk keamanan.")
                        self.close_position(current_price, "SL_UPDATE_FAIL")
                        return

            else: # SHORT
                # SL baru harus lebih rendah dari SL sebelumnya
                # Hitung SL baru berdasarkan current_price dan trailing_rr
                new_sl_price = self._round_price(current_price + (current_atr * self.trailing_rr))
                if new_sl_price < current_sl_internal:
                    print(f"INFO: Mengupdate Trailing SL untuk SHORT. Old SL: {current_sl_internal} -> New SL: {new_sl_price}")
                    self._cancel_order(self.position['sl_order_id'], self.symbol)
                    try:
                        new_sl_order_id = self._place_sl_order(position_type, qty, new_sl_price)
                        self.position['sl'] = new_sl_price
                        self.position['sl_order_id'] = new_sl_order_id
                    except Exception as e:
                        print(f"CRITICAL ERROR: Gagal menempatkan ulang SL order saat trailing: {e}. Menutup posisi untuk keamanan.")
                        self.close_position(current_price, "SL_UPDATE_FAIL")
                        return

    def close_position_after_fill(self, exit_type, current_price):
        """Dipanggil setelah SL/TP order terisi di Binance."""
        if self.position['status'] == 'NONE':
            return # Sudah ditutup

        entry_price = self.position['entry_price']
        qty = self.position['qty']
        
        # PnL hanya perkiraan, PnL sebenarnya harus diambil dari statement Binance
        pnl = (current_price - entry_price) * qty if self.position['type'] == 'LONG' else (entry_price - current_price) * qty
        print(f"INFO: Posisi {self.position['type']} ditutup (Type: {exit_type}). PnL (Est.): {pnl:.4f}")
        self._reset_position_state()


    def close_position(self, exit_price, exit_type):
        """Menutup posisi secara paksa dengan market order."""
        if self.position['status'] == 'NONE':
            print("ERROR: Tidak ada posisi aktif untuk ditutup.")
            return

        side = 'SELL' if self.position['type'] == 'LONG' else 'BUY' # side to close the position
        qty = self.position['qty']
        entry_price = self.position['entry_price']

        print(f"INFO: Menutup posisi {self.position['type']} secara paksa ({exit_type})...")
        
        # Batalkan semua order terbuka terkait posisi ini
        self._cancel_order(self.position['sl_order_id'], self.symbol)
        self._cancel_order(self.position['tp_order_id'], self.symbol)

        try:
            # Menutup posisi dengan market order
            close_order = self.exchange.create_market_order(self.symbol, side, qty, {'reduceOnly': True})
            
            pnl = (exit_price - entry_price) * qty if self.position['type'] == 'LONG' else (entry_price - exit_price) * qty
            
            print(f"INFO: Posisi {self.position['type']} ditutup pada {exit_price} (Type: {exit_type}). PnL (Est.): {pnl:.4f}")
            self._reset_position_state()

        except Exception as e:
            print(f"CRITICAL ERROR: Gagal menutup posisi {self.position['type']} secara paksa: {e}")
            # Jika gagal menutup posisi secara paksa, ada masalah serius. Perlu notifikasi manual.

    def _reset_position_state(self):
        """Mereset status posisi bot."""
        self.position = {
            'status': 'NONE',
            'type': None,
            'entry_price': None,
            'qty': 0,
            'sl': None,
            'tp': None,
            'risk_amount': 0,
            'sl_order_id': None,
            'tp_order_id': None,
            'is_asia_entry': None,
            'entry_time': None
        }

    def run(self):
        print(f"INFO: Bot Supertrend mulai berjalan untuk {self.symbol} pada timeframe {self.timeframe}...")
        
        while True:
            try:
                # 1. Fetch current price
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']

                # 2. Ambil data OHLCV terbaru dari Binance
                # Ambil cukup data untuk semua indikator (EMA200 + ATR period + buffer)
                ohlcv_limit = self.atr_period + 200 + 10 # Buffer for indicator calculation
                df = self.fetch_ohlcv(limit=ohlcv_limit) 
                
                if df.empty or len(df) < ohlcv_limit:
                    print("WARNING: Data OHLCV tidak cukup. Menunggu data lebih banyak...")
                    time.sleep(60) # Tunggu sebentar sebelum mencoba lagi
                    continue

                # Pastikan ATR dihitung pertama karena digunakan di add_indicators
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=self.atr_period)
                
                # Lanjutkan dengan perhitungan indikator
                df = self.calculate_supertrend(df) # Ini akan mencoba dua cara untuk supertrend
                
                # Penting: Check if 'ST' column was successfully added by calculate_supertrend
                if 'ST' not in df.columns:
                    print("CRITICAL ERROR: Supertrend calculation failed. Skipping this iteration.")
                    time.sleep(60)
                    continue

                df = self.add_indicators(df)
                df = self.apply_time_filters(df)

                # Ambil data terbaru setelah semua indikator dihitung
                last_candle = df.iloc[-1]
                current_atr = last_candle['ATR']
                
                # Pastikan current_atr valid
                if pd.isna(current_atr) or current_atr == 0:
                    print("WARNING: ATR tidak valid di candle terakhir. Menunggu data berikutnya.")
                    time.sleep(60)
                    continue

                print(f"INFO: {datetime.now()} - Harga: {current_price:.{self.price_decimals}f} ATR: {current_atr:.{self.price_decimals}f}")

                # Kelola posisi yang sudah ada
                if self.position['status'] == 'OPEN':
                    self.manage_position(current_price, current_atr) # Pass current_price & atr untuk trailing
                
                # Cek sinyal baru jika tidak ada posisi aktif
                else:
                    long_signal, short_signal, trade_info = self.check_signals(df)
                    
                    if long_signal:
                        print("INFO: Sinyal LONG terdeteksi!")
                        # Hitung SL price awal (berdasarkan RR_SL_Initial)
                        sl_price = trade_info['entry_price'] - (trade_info['atr_value'] * trade_info['rr_sl_initial'])
                        sl_price = self._round_price(sl_price)
                        
                        # Hitung TP price (berdasarkan RR_TP_Fixed)
                        tp_price = trade_info['entry_price'] + ((trade_info['entry_price'] - sl_price) * trade_info['rr_tp_fixed'])
                        tp_price = self._round_price(tp_price)
                        
                        qty = self.calculate_position_sizes(trade_info['entry_price'], trade_info['atr_value'], trade_info['rr_sl_initial'])
                        
                        if qty == 0:
                            print("WARNING: Calculated quantity is zero. Aborting trade.")
                            continue

                        # Place entry order (LIMIT)
                        entry_order = self.place_entry_order('LONG', qty, trade_info['entry_price'])
                        if entry_order and entry_order['status'] == 'open':
                            print(f"INFO: Menunggu entry LIMIT order {entry_order['id']} terisi...")
                            time.sleep(5) # Beri waktu agar order terisi (bisa disesuaikan)
                            filled_order = self.exchange.fetch_order(entry_order['id'], self.symbol)
                            if filled_order['status'] == 'closed' and filled_order['filled'] > 0:
                                # Update status posisi internal dengan data order yang terisi
                                self.position.update({
                                    'status': 'OPEN',
                                    'type': 'LONG',
                                    'entry_price': self._round_price(filled_order['price']), # Gunakan harga terisi
                                    'qty': self._round_qty(filled_order['filled']),          # Gunakan qty terisi
                                    'sl': sl_price, # SL awal
                                    'tp': tp_price, # TP awal
                                    'risk_amount': self.base_risk,
                                    'is_asia_entry': trade_info['is_asia_entry'],
                                    'entry_time': datetime.now()
                                })
                                print(f"INFO: Entry order {entry_order['id']} terisi. Entry price actual: {self.position['entry_price']}, Qty actual: {self.position['qty']}")
                                # Place SL (STOP_MARKET) and TP (LIMIT) orders
                                try:
                                    sl_id, tp_id = self.place_stop_loss_take_profit_orders('LONG', self.position['qty'], self.position['sl'], self.position['tp'])
                                    self.position['sl_order_id'] = sl_id
                                    self.position['tp_order_id'] = tp_id
                                except Exception as e:
                                    print(f"CRITICAL ERROR: Gagal menempatkan SL/TP setelah entry: {e}. Mencoba menutup posisi...")
                                    self.close_position(current_price, "SL_TP_PLACE_FAIL")
                            else:
                                print(f"WARNING: Entry LIMIT order {entry_order['id']} tidak terisi sepenuhnya atau dibatalkan. Mereset posisi.")
                                self._cancel_order(entry_order['id'], self.symbol) # Pastikan entry order dibatalkan
                                self._reset_position_state()
                        else:
                            print("ERROR: Gagal menempatkan entry order atau order tidak terisi. Mereset posisi.")
                            self._reset_position_state()


                    elif short_signal:
                        print("INFO: Sinyal SHORT terdeteksi!")
                        # Hitung SL price awal (berdasarkan RR_SL_Initial)
                        sl_price = trade_info['entry_price'] + (trade_info['atr_value'] * trade_info['rr_sl_initial'])
                        sl_price = self._round_price(sl_price)
                        
                        # Hitung TP price (berdasarkan RR_TP_Fixed)
                        tp_price = trade_info['entry_price'] - ((sl_price - trade_info['entry_price']) * trade_info['rr_tp_fixed'])
                        tp_price = self._round_price(tp_price)
                        
                        qty = self.calculate_position_sizes(trade_info['entry_price'], trade_info['atr_value'], trade_info['rr_sl_initial'])

                        if qty == 0:
                            print("WARNING: Calculated quantity is zero. Aborting trade.")
                            continue

                        # Place entry order (LIMIT)
                        entry_order = self.place_entry_order('SHORT', qty, trade_info['entry_price'])
                        if entry_order and entry_order['status'] == 'open':
                            print(f"INFO: Menunggu entry LIMIT order {entry_order['id']} terisi...")
                            time.sleep(5) # Beri waktu agar order terisi (bisa disesuaikan)
                            filled_order = self.exchange.fetch_order(entry_order['id'], self.symbol)
                            if filled_order['status'] == 'closed' and filled_order['filled'] > 0:
                                # Update status posisi internal dengan data order yang terisi
                                self.position.update({
                                    'status': 'OPEN',
                                    'type': 'SHORT',
                                    'entry_price': self._round_price(filled_order['price']), # Gunakan harga terisi
                                    'qty': self._round_qty(filled_order['filled']),          # Gunakan qty terisi
                                    'sl': sl_price, # SL awal
                                    'tp': tp_price, # TP awal
                                    'risk_amount': self.base_risk,
                                    'is_asia_entry': trade_info['is_asia_entry'],
                                    'entry_time': datetime.now()
                                })
                                print(f"INFO: Entry order {entry_order['id']} terisi. Entry price actual: {self.position['entry_price']}, Qty actual: {self.position['qty']}")
                                # Place SL (STOP_MARKET) and TP (LIMIT) orders
                                try:
                                    sl_id, tp_id = self.place_stop_loss_take_profit_orders('SHORT', self.position['qty'], self.position['sl'], self.position['tp'])
                                    self.position['sl_order_id'] = sl_id
                                    self.position['tp_order_id'] = tp_id
                                except Exception as e:
                                    print(f"CRITICAL ERROR: Gagal menempatkan SL/TP setelah entry: {e}. Mencoba menutup posisi...")
                                    self.close_position(current_price, "SL_TP_PLACE_FAIL")
                            else:
                                print(f"WARNING: Entry LIMIT order {entry_order['id']} tidak terisi sepenuhnya atau dibatalkan. Mereset posisi.")
                                self._cancel_order(entry_order['id'], self.symbol) # Pastikan entry order dibatalkan
                                self._reset_position_state()
                        else:
                            print("ERROR: Gagal menempatkan entry order atau order tidak terisi. Mereset posisi.")
                            self._reset_position_state()
                    else:
                        # print(f"INFO: Tidak ada sinyal baru.") # Ter