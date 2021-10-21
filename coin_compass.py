import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from PIL import Image
import codecs
import streamlit.components.v1 as stc


st.sidebar.image("logo.jpg", use_column_width=True)
st.sidebar.header("NAVIGATION")
rad = st.sidebar.radio("<3", ["Home","Dashboard","Coin Forecast","USD Forecast"])

#Home Page------------------------------------------------------------------------------
if rad == "Home":
	img = Image.open("f3.jpg")
	st.image(img)
	st.title("Home")

	col1, col2 = st.columns(2)
	with col1:
		st.subheader('What is Coin Compass?')
		"Coin Compass is a versitile web application which makes use of a sophisticated high powered machine learning model for analyis and forecast of Bitcoina nd five(5) other cryptocurrencies on the market."
		"Coin Compass generates forecast data with high speed and great accuracy! It provides real time prices of coins and give you the opportunity to visualize and compare price trends of available coins."

	with col2:
		st.subheader(' ')
		"Coin Compass makes use of Facebook Prophet, a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects."
		"It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well."

#Dashboard ------------------------------------------------------------------------
if rad == "Dashboard":
	img = Image.open("f1.jpg")
	st.image(img)
	st.title("Coin Compasss Dashboard")
	

	coin = ("BTC-USD", "ETH-USD","XLM-USD", "LTC-USD","LINK-USD", "ADA-USD")

	dropdown = st.multiselect('Select Coin(s)', coin)

	start = st.date_input('Start', value=pd.to_datetime('2021-01-01'))
	end = st.date_input('End', value=pd.to_datetime('today'))

	if len(dropdown) > 0:
		df = yf.download(dropdown,start,end)['Close']

		
		st.header("Graph(s) Of Closing Price(s) of {}".format(dropdown))
		st.line_chart(df)

# Forecast Section/Page------------------------------------------------------------

if rad == "Coin Forecast":
	img = Image.open("f2.jpg")
	st.image(img)
	st.title("Forecast")
	

	START = "2017-01-01"
	TODAY = date.today().strftime("%Y-%m-%d")

	coin = ("BTC-USD", "ETH-USD","XLM-USD", "LTC-USD","LINK-USD", "ADA-USD")
	select_coin = st.selectbox("Select coin for prediction", coin)

	n_years = st.slider("Years of Prediction:", 1, 365)
	period = n_years

	@st.cache
	def load_data(ticker):
		data = yf.download(ticker, START, TODAY)
		data.reset_index(inplace = True)
		return data

	data_load_state = st.text("Load data...")
	data = load_data(select_coin)
	data_load_state.text("Loading data...done!")

	st.subheader('Raw data')
	st.write(data.tail())

	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='opening_prices'))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='closing_prices'))
		fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True)
		st.plotly_chart(fig)

	plot_raw_data()

	#df_train = data[['Date', 'Close']]
	#df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	#Feature Selection

	coin_close = data.drop(data.columns[[1,2,3,5,6]], axis=1)
	coin_vol = data.drop(data.columns[[1,2,3,4,5]], axis=1)
	coin_close = coin_close.rename(columns={"Date": "ds", "Close": "y"})
	coin_vol = coin_vol.rename(columns={"Date":"ds", "Volume":"y"})
	coin_close.head()

	# Forecasting closing prices
	
	m = Prophet(daily_seasonality=True)
	m.fit(coin_close)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)
	st.header('Forecast Closing Prices')
	st.write(forecast.tail())


	#Graphs of Forecasted Closing Price Data
	st.header('Forecast Closing Prices Graph')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.subheader('Closing Price Forecast Components')
	fig2 = m.plot_components(forecast)
	st.write(fig2)

	# Forecasting Volume
	
	m = Prophet(daily_seasonality=True)
	m.fit(coin_vol)
	future1 = m.make_future_dataframe(periods=period)
	forecast1 = m.predict(future1)
	st.header('Forecast Volume')
	st.write(forecast1.tail())

	#Graphs of Forecasted Volume
	st.header('Forecast Volume Graph')
	fig3 = plot_plotly(m, forecast1)
	st.plotly_chart(fig3)

	st.subheader('Volume Forecast Components')
	fig4 = m.plot_components(forecast1)
	st.write(fig4)

#USD Forecast Page-------------------------------------------------------------------

if rad == "USD Forecast":
	img = Image.open("f1.jpg")
	st.image(img)
	st.title("USD Forecast")
	

	START = "2017-01-01"
	TODAY = date.today().strftime("%Y-%m-%d")

	n_years = st.slider("Years of Prediction:", 1, 365)
	period = n_years

	#Load Dataset
	usd = pd.read_csv("https://github.com/VanessaAttaFynn/Final_Year_Project/tree/main/data/USD.csv",parse_dates=['Date'])
	#Feature Selection
	usd_price = usd.drop(usd.columns[[2,3,4,5,6]], axis=1)
	usd_price = usd_price.rename(columns={"Date": "ds", "Price": "y"})
	usd_price.head()

	# Forecasting USD
	m = Prophet(daily_seasonality=True)
	m.fit(usd_price)
	future2 = m.make_future_dataframe(periods=period)
	forecast2 = m.predict(future2)
	st.header('Forecast USD Prices')
	st.write(forecast2.tail())

	#Graphs of Forecasted USD
	st.header('Forecast USD Graph')
	fig5 = plot_plotly(m, forecast2)
	st.plotly_chart(fig5)

	st.subheader('USD Forecast Components')
	fig6 = m.plot_components(forecast2)
	st.write(fig6)

	#Select coin to view prediction

	coin = ("BTC-USD", "ETH-USD","XLM-USD", "LTC-USD","LINK-USD", "ADA-USD")
	select_coin = st.selectbox("Select coin for prediction", coin)

	@st.cache
	def load_data(ticker):
		data = yf.download(ticker, START, TODAY)
		data.reset_index(inplace = True)
		return data

	data_load_state = st.text("Load data...")
	data = load_data(select_coin)
	data_load_state.text("Loading data...done!")

	#Feature Selection

	coin_close = data.drop(data.columns[[1,2,3,5,6]], axis=1)
	coin_close = coin_close.rename(columns={"Date": "ds", "Close": "y"})
	coin_close.head()

	# Forecasting closing prices
	
	m = Prophet(daily_seasonality=True)
	m.fit(coin_close)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)
	st.header('Forecast Data')
	st.write(forecast.tail())


	#Graphs of Forecasted Closing Price Data
	st.header('Forecast Data Graph')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.subheader('Forecast Components')
	fig2 = m.plot_components(forecast)
	st.write(fig2)

