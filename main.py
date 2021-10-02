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
rad = st.sidebar.radio("NAVIGATION", ["Home","Dashboard","Forecast"])


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

# Forecast Section/Page

if rad == "Forecast":
	img = Image.open("f2.jpg")
	st.image(img)
	st.title("Forecast")
	

	START = "2020-01-01"
	TODAY = date.today().strftime("%Y-%m-%d")

	coin = ("BTC-USD", "ETH-USD","XLM-USD", "LTC-USD","LINK-USD", "ADA-USD")
	select_coin = st.selectbox("Select coin for prediction", coin)

	n_years = st.slider("Years of Prediction:", 1, 4)
	period = n_years*365

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

	#Forecasting
	df_train = data[['Date', 'Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)
	st.subheader('Forecast Data')
	st.write(forecast.tail())


#Graphs of Forecasted Data
	st.write('Forecast Data Graph')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.write('Forecast Components')
	fig2 = m.plot_components(forecast)
	st.write(fig2)