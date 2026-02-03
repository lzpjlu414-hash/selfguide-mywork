eggs_per_day(16).
breakfast_eggs(3).
muffin_eggs(4).
price_per_egg(2).
daily_profit(D) :- eggs_per_day(E), breakfast_eggs(B), muffin_eggs(M), price_per_egg(P), D is (E - B - M) * P.
daily_profit(D).
