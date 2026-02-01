eggs_per_day(16).
breakfast_eggs(3).
muffin_eggs(4).
price_per_egg(2).
daily_earnings(Earnings) :- eggs_per_day(Total), breakfast_eggs(Breakfast), muffin_eggs(Muffin), price_per_egg(Price), Remaining is Total - Breakfast - Muffin, Earnings is Remaining * Price.
daily_earnings(Answer).
