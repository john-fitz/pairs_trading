# pairs_trading

This project was an attempt at building a functional pairs trading platform for cryptocurrency using the Binance US API. The goal was to gather information on potential coins, identify which were suitable for pairs trading, and then trade them with some entry/exit conditions with a simple backtesting platform. If successful in backtesting, it would have been bundled and put on a cloud server to run continuously. Instead, it's here for everyone to enjoy.

I used the Binance API because of a friend's suggestion and it's ease-of-use while collecting historical information and Python just due to my comfort level and ability to do all in one. This could have been created with another trading platform (in fact, for reasons stated later, it would have needed to be).

I was inspired to work on the project after learning about it's ubiquity in the algorithmic trading space and reading Vidyamurthy's "Pairs Trading: Quantitative Methods and Analysis" and wondering how difficult it would be to convert the theory into practice. As it turns out, the infrastructure just to start on something like this is a heavy lift for one person. I followed the motto "work fast and break things", but I also realized that for somehting as large as this and that carried a financial risk, that might not be as smart.

I found minor bugs (that I hopefully fixed most of) and it performed moderately well in backtesting, but the biggest issue was clear: I could not actually short many of the coins that I wanted to. The financial isntruments just did not exist. And since pairs trading requires simultaneous long and short positions in different coins, it was not going to be feasible to implement this project. Towards the end, I tried to build a long portfolio of potential coins and create pseudo short positions by selling and buying back positions in my own portfolio, but I did not have much success - and I didn't want to be exposed to crypto currency swings in my portfolio.

Overall, it was an interesting project and I learned a lot about how to build a piece of software as large and complicated as it was. For those wondering in case they wanted to build something similar: the approximate  breakdown for me to build this was 15% pairs trading/stats knowledge, 70% software engineering, 10% general finance knowledge, 5% banging my head against a wall.

If you want to try out the project (which I suggest poking around a little before you run it), feel free to download the files and most of the logic is run through bot_script.py except for the historical information which is downloaded and saved in a CSV file using the create_and_save_historicals function in pairs_helpers.py. Once you have that, you can run the backtesting by opening terminal, navigating to the project file, and typing "python bot_script.py". It takes up a lot of processing power and memory to run it because it's testing and recompiling lists of potential trades every hour, so I've tried to optimize it at different times to cut down on backtesting time and there are a lot of CSV files that are created and saved in the process. I suggest checking out past commits to see which steps you need to take in order to get it running properly. 

MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
