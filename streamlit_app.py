import plotly.express as px
import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np

st.set_page_config(layout='wide')

st.title('🎈Streamlit Project')

url_input = 'https://raw.githubusercontent.com/Rock-Atikhom/Test/master/test_data.csv'

##st.subheader('Output')
##st.warning(f'The URL of your dataset is: {url_input}')

if url_input:
    with st.expander('Dataset Preview'): 
        df = pd.read_csv(url_input)
        # st.subheader('Dataset Preview')
        st.dataframe(df, use_container_width=True)

    with st.expander('Check Missing Value'):
        ## st.subheader('Check Missing Value')
        missing_value = df.isna().sum()
        st.dataframe(missing_value)

    with st.expander('Re-Size and Replace Column'):
        ## st.subheader('Re-size and Replace Column')
        re_col = [col.lower().replace(" ", "_") for col in df.columns]
        df.columns = re_col
        st.dataframe(df.head(11), use_container_width=True)

    with st.expander('Detect Outliers'):
        # st.subheader('Detect Outliers')
        st.dataframe(df.describe()[['price', 'kitchen_staff', 'drinks_staff']], use_container_width=True)

    with st.expander('Change Object to Datetime'):
        # st.subheader('Change Object to Datetime')
        df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
        df['order_time'] = pd.to_datetime(df['order_time'], format = 'ISO8601')
        df['serve_time'] = pd.to_datetime(df['serve_time'], format = 'ISO8601')
        st.dataframe(df[['date', 'order_time', 'serve_time']].head(11), use_container_width=True)

    with st.expander('Menu Name'):
    # st.subheader('Menu Name')
        st.write(list(df['menu'].unique()))

    ## 01
    # st.header('Quantity by Menu')
    order_quantity = df[['menu', 'category']]\
    .groupby('menu').agg(total_menu = ('menu','count'))\
    .sort_values('total_menu', ascending=False).reset_index()
    # st.bar_chart(order_quantity, y='total_menu', x='menu',use_container_width=True, color='#ffe599')

    chart_order_quantity = px.bar(
        order_quantity, 
        x='total_menu', y='menu',
        orientation='h', color='menu', title = 'Overall Quantity by Menu')
    # st.plotly_chart(chart_order_quantity)

    chart_order_quantity.update_layout(
        title_text='Overall Quantity by Menu',
        xaxis_title='Quantity',
        yaxis_title='Menu')
    st.plotly_chart(chart_order_quantity, use_container_width=True)

    ## 02
    ## category quantity count
    ## drink > food
    category_quantity = df[['menu','category']].groupby('category')['category'].agg(total_category = 'count').reset_index()
    pie_category = px.pie(
        category_quantity, 
        values='total_category', names='category', 
        title='Proportion by Category')
    st.plotly_chart(pie_category, use_container_width=True)
    
    ## 03
    # st.header('Overall Sales by Food Category')
    revenue_food = df.query("category == 'food' ")\
    .groupby(['menu', 'category', 'price'])['price']\
    .agg(total_price = ('sum')).sort_values('total_price', ascending = False).reset_index()
    # st.bar_chart(revenue_food, x='menu', y='total_price', color='menu')
    
    chart_price_food = px.bar(
        revenue_food, 
        x='total_price', y='menu',
        orientation='h', color='menu', title = 'Overall Sales by Food Category')

    chart_price_food.update_layout(
        title_text='Overall Sales by Food Category',
        xaxis_title='Sales',
        yaxis_title='Menu')
    st.plotly_chart(chart_price_food, use_container_width=True)

    ## 04
    # st.header('Overall Sales by Drink Category')
    revenue_drink = df.query("category == 'drink' ")\
    .groupby(['menu', 'category', 'price'])['price']\
    .agg(total_price = ('sum')).sort_values('total_price', ascending = False).reset_index()
    # st.bar_chart(revenue_drink, x='menu', y='total_price', color='menu')

    chart_price_drink = px.bar(
        revenue_drink, 
        x='total_price', y='menu',
        orientation='h', color='menu', title = 'Overall Sales by Drink Category')
    
    chart_price_drink.update_layout(
        title_text='Overall Sales by Drink Category',
        xaxis_title='Sales',
        yaxis_title='Menu')
    st.plotly_chart(chart_price_drink, use_container_width=True)


    ## 05
    ## kitchen_staff
    # df2 = df.query("category == 'food'")
    # df2['date'] = df2['date'].dt.strftime('%m')

    # ## drinks_staff
    # df3 = df.query("category == 'drink'")
    # df3['date'] = df3['date'].dt.strftime('%m')

    ## enter the store by hour
    df3 = df
    df3['hours'] = df3['order_time'].dt.strftime('%H')
    
    # st.header('Consumer Behavior by Hour')
    trend_consumer = df3[['hours', 'menu']]\
    .groupby('hours')[['hours','menu']]\
    .agg(trend_consumer = ('hours', 'count')).reset_index()
    # st.line_chart(trend_consumer, x='hours', y='trend_consumer')

    hour_timeline = px.line(
        trend_consumer, 
        x='hours', y='trend_consumer',
        orientation='h', title = 'Consumer Behavior by Hour', line_shape='linear')

    hour_timeline.update_traces(textposition='top center')
    hour_timeline.update_layout(
        title_text='Consumer Behavior by Hour',
        xaxis_title='Hour',
        yaxis_title='Consumer')
    st.plotly_chart(hour_timeline, use_container_width=True)


    ## 06
    ## enter the store by day of week
    ## st.header('Consumer Behavior by Day of Week')
    df3['week_of_days'] = df3['date'].dt.strftime('%w')

    # weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # df3['day_of_week'] = pd.Categorical(df3.day_of_week,categories=weekdays)

    day_of_weeks = df3[['week_of_days', 'menu']]\
    .groupby('week_of_days')[['week_of_days','menu']]\
    .agg(trend_consumer = ('week_of_days', 'count')).sort_values('week_of_days').reset_index()
    # st.line_chart(day_of_weeks, x='week_of_days',y='trend_consumer')

    day_timeline = px.line(
        day_of_weeks, 
        x='week_of_days', y='trend_consumer',
        orientation='v', title = 'Consumer Behavior by Day of Week', line_shape='linear')
    
    day_timeline.update_layout(
        title_text='Consumer Behavior by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Consumer')
    st.plotly_chart(day_timeline, use_container_width=True)


    ## 07
    ## kitchen and drinks staff
    df2 = df
    df2['date'] = df2['date'].dt.strftime('%m')

    ## st.header('Kitchen and Drinks staff by month')
    kitchen_drinks_month = df2[['date', 'category', 'kitchen_staff', 'drinks_staff', 'menu']]\
    .groupby('date')[['kitchen_staff', 'drinks_staff', 'menu']]\
    .agg(avg_kitchen_staff = ('kitchen_staff','mean'), avg_drinks_staff = ('drinks_staff', 'mean') , total_order = ('menu', 'count'))\
    .sort_values('date', ascending=True).reset_index()
    ## st.bar_chart(kitchen_drinks_month, x='date', y=['avg_drinks_staff', 'avg_kitchen_staff'])

    kitchen_month = px.bar(
        kitchen_drinks_month, 
        y=['avg_drinks_staff', 'avg_kitchen_staff'], x='date',
        orientation='v', title = 'Kitchen and Drinks staff by Month', barmode='group')
    
    kitchen_month.update_layout(
        title_text='Kitchen and Drinks Staff by Month',
        xaxis_title='Month',
        yaxis_title='Average Staff')
    st.plotly_chart(kitchen_month, use_container_width=True)

    
    ## 08
    ## st.header('Kitchen and Drinks staff by Day of Week')
    ## kitchen and drinks staff by day
    kitchen_drinks_day = df3[['week_of_days', 'category', 'kitchen_staff', 'drinks_staff', 'menu']]\
    .groupby('week_of_days')[['kitchen_staff', 'drinks_staff', 'menu']]\
    .agg(avg_kitchen_staff = ('kitchen_staff','mean'), avg_drinks_staff = ('drinks_staff', 'mean') , total_order = ('menu', 'count'))\
    .sort_values('week_of_days', ascending=True).reset_index()
    ## st.bar_chart(kitchen_drinks_month, x='date', y=['avg_drinks_staff', 'avg_kitchen_staff'])

    kitchen_day = px.bar(
        kitchen_drinks_day,
        y=['avg_drinks_staff','avg_kitchen_staff'], x='week_of_days',
        orientation='v', title='Kitchen and Drinks Staff by Day of Week', barmode='group')
    # st.plotly_chart(kitchen_day)

    kitchen_day.update_layout(
        title_text='Kitchen and Drinks Staff by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Average Staff')
    st.plotly_chart(kitchen_day, use_container_width=True)


    st.write('**Food by Kitchen Staff**')
    ## change datetime to hours for food and drink category
    df3['start_time'] = df['order_time'].dt.strftime('%H:%M:%S')
    df3['finish_time'] = df['serve_time'].dt.strftime('%H:%M:%S')

    df3['start_time'] = pd.to_datetime(df3['start_time'], format= '%H:%M:%S')
    df3['finish_time'] = pd.to_datetime(df3['finish_time'], format= '%H:%M:%S')

    ## cooking time for food and drink category
    df3['cooking_time'] = df3['finish_time'] - df3['start_time']

    ## cooking time for kitchen staff
    cooking_food = df3[['menu', 'category', 'kitchen_staff', 'cooking_time']].query("category == 'food'")\
    .groupby(['menu', 'category'])[['menu', 'kitchen_staff', 'cooking_time']]\
    .agg(avg_kitchen = ('kitchen_staff', 'mean'), avg_cooking_time = ('cooking_time', 'mean') ,total_order = ('menu', 'count'))\
    .sort_values('total_order', ascending = False).reset_index()
    # st.line_chart(cooking_food, x='avg_cooking_time', '')
    st.dataframe(cooking_food, use_container_width=True)


    st.write('**Drink by Drinks Staff**')
    cooking_drink = df3[['menu', 'category', 'drinks_staff', 'cooking_time']].query("category == 'drink'")\
    .groupby(['menu', 'category'])[['menu', 'drinks_staff', 'cooking_time']]\
    .agg(avg_drinks_staff = ('drinks_staff', 'mean'), avg_cooking_time = ('cooking_time', 'mean') ,total_order = ('menu', 'count'))\
    .sort_values('total_order', ascending = False).reset_index()
    st.dataframe(cooking_drink, use_container_width=True)

else:
    st.error('⬅️ Awaiting your Dataset!')
