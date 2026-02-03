eggs_per_day(16).
breakfast_eggs(3).
muffin_eggs(4).
price_per_egg(2).
daily_profit(Profit) :- eggs_per_day(Total), breakfast_eggs(Breakfast), muffin_eggs(Muffin), price_per_egg(Price), Remaining is Total - Breakfast - Muffin, Profit is Remaining * Price.
daily_profit(Profit).
