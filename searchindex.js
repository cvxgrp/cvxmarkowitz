Search.setIndex({"docnames": ["docs/api", "docs/index", "docs/nb", "docs/notebooks/capacity", "docs/notebooks/demo", "docs/notebooks/difference", "docs/notebooks/monkey", "docs/notebooks/pairs", "docs/notebooks/quantstats", "docs/reports"], "filenames": ["docs/api.md", "docs/index.md", "docs/nb.md", "docs/notebooks/capacity.ipynb", "docs/notebooks/demo.ipynb", "docs/notebooks/difference.ipynb", "docs/notebooks/monkey.ipynb", "docs/notebooks/pairs.ipynb", "docs/notebooks/quantstats.ipynb", "docs/reports.md"], "titles": ["API", "Simulator", "Notebooks", "Estimating capacity using boxing constraints", "Long only 1/n portfolio", "Portfolio difference", "Monkey portfolios", "Almost pairs trading", "With quantstats", "ci/cd reports"], "terms": {"given": [1, 3, 4], "univers": [1, 7], "m": [1, 4], "asset": [1, 3, 4, 5, 6, 7, 8], "we": [1, 3, 4, 8], "ar": [1, 3, 4], "price": [1, 3, 5, 6, 7, 8], "each": [1, 4, 7, 8], "them": 1, "t_1": 1, "t_2": 1, "ldot": 1, "t_n": 1, "e": [1, 4], "g": [1, 4], "oper": 1, "us": [1, 2], "an": [1, 3, 7], "n": [1, 2, 6, 8], "matrix": 1, "where": [1, 4], "column": [1, 4], "correspond": 1, "particular": 1, "In": 1, "backtest": 1, "iter": 1, "row": [1, 4], "alloc": [1, 3], "posit": [1, 3, 4, 5], "all": [1, 3], "some": [1, 3], "thi": [1, 4, 7], "tool": 1, "shall": 1, "help": [1, 4], "simplifi": 1, "account": 1, "It": 1, "keep": 1, "track": 1, "avail": 1, "cash": [1, 5], "profit": [1, 4], "achiev": 1, "etc": 1, "The": [1, 7], "complet": [1, 4, 7], "agnost": [1, 7], "trade": [1, 2, 3], "polici": [1, 7], "strategi": [1, 7], "our": 1, "approach": [1, 4], "follow": 1, "rather": [1, 4], "common": 1, "pattern": 1, "demonstr": [1, 7], "those": 1, "step": 1, "somewhat": 1, "silli": [1, 7], "thei": 1, "never": 1, "good": [1, 7], "alwai": [1, 7], "valid": [1, 7], "ones": [1, 5], "user": 1, "defin": [1, 4], "load": [1, 7], "frame": [1, 4], "initi": 1, "amount": [1, 7], "experi": 1, "from": [1, 3, 4, 5, 6, 7, 8], "pathlib": [1, 7], "import": [1, 3, 4, 5, 6, 7, 8], "path": [1, 7], "panda": [1, 3, 4, 5, 6, 7, 8], "pd": [1, 3, 4, 5, 6, 7, 8], "cvx": [1, 3, 4, 5, 6, 7, 8], "read_csv": [1, 3, 5, 6, 7, 8], "resourc": 1, "csv": [1, 3, 5, 6, 7, 8], "index_col": [1, 3, 5, 6, 7, 8], "0": [1, 3, 4, 5, 6, 7, 8], "parse_d": [1, 3, 5, 6, 7, 8], "true": [1, 3, 4, 5, 6, 7, 8], "header": [1, 3, 5, 6, 7, 8], "ffill": [1, 7], "b": [1, 3, 4, 5, 6, 7, 8], "initial_cash": [1, 3, 4, 5, 6, 7, 8], "1e6": [1, 3, 4, 5, 6, 7, 8], "also": 1, "possibl": 1, "specifi": 1, "model": [1, 7], "cost": [1, 7], "fill": [1, 4], "up": 1, "onli": [1, 2], "onc": [1, 4], "done": 1, "construct": [1, 4], "actual": 1, "portfolio": [1, 2, 3, 7, 8], "have": [1, 3, 4], "overload": 1, "__iter__": 1, "__setitem__": 1, "method": 1, "custom": 1, "let": [1, 7], "s": [1, 7], "start": 1, "first": 1, "dai": [1, 7, 8], "choos": [1, 7], "two": [1, 3, 4, 7], "name": [1, 7], "random": [1, 5, 6, 7], "bui": [1, 7], "one": [1, 7], "sai": [1, 7], "1": [1, 2, 3, 5, 6, 7, 8], "your": [1, 4, 7], "wealth": [1, 7], "short": [1, 5, 7], "same": [1, 7], "t": [1, 3, 4, 7, 8], "state": [1, 4, 5, 6, 7, 8], "pick": [1, 3, 7], "pair": [1, 2], "np": [1, 5, 6, 7], "choic": [1, 7], "2": [1, 3, 4, 7], "replac": [1, 7], "fals": [1, 4, 7], "comput": [1, 7], "stock": [1, 4, 7], "seri": [1, 5, 6, 7], "index": [1, 4, 5, 6, 7], "data": [1, 3, 4, 5, 6, 7, 8], "nav": [1, 3, 4, 5, 6, 7, 8], "valu": [1, 7], "updat": 1, "here": [1, 7], "grow": 1, "list": [1, 4], "timestamp": 1, "t1": 1, "t2": 1, "second": 1, "t3": 1, "A": [1, 3], "lot": 1, "magic": 1, "hidden": 1, "variabl": 1, "give": 1, "access": 1, "current": 1, "valuat": 1, "hold": 1, "slightli": 1, "more": 1, "realist": 1, "set": 1, "4": [1, 3, 4], "want": 1, "implmen": 1, "popular": 1, "invest": [1, 4, 8], "quarter": [1, 4, 8], "capit": [1, 3, 4, 8], "25": [1, 3, 4], "note": 1, "last": 1, "element": 1, "than": 1, "weight": [1, 3, 5], "cashposit": 1, "class": 1, "expos": 1, "setter": 1, "convent": 1, "set_weight": [1, 5], "finish": 1, "build": [1, 3, 4, 5, 6, 7, 8], "abov": 1, "desir": [1, 3], "after": 1, "trigger": 1, "readi": 1, "further": 1, "analysi": 1, "dive": 1, "equiti": 1, "mai": [1, 3], "know": 1, "enter": [1, 3], "etern": 1, "run": 1, "non": 1, "python": 1, "wast": 1, "case": 1, "submit": 1, "togeth": 1, "when": 1, "equityportfolio": 1, "assum": [1, 3], "you": [1, 4, 7], "share": [1, 4], "alreadi": 1, "love": 1, "instal": 1, "can": [1, 4], "perform": 1, "replic": 1, "virtual": 1, "environ": 1, "pyproject": 1, "toml": 1, "jupyterlab": 1, "within": [1, 4], "new": 1, "execut": [1, 4], "create_kernel": 1, "sh": 1, "dedic": 1, "project": 1, "estim": 2, "capac": 2, "box": 2, "constraint": 2, "long": [2, 5], "differ": [2, 4], "monkei": 2, "almost": [2, 4], "With": 2, "quantstat": 2, "simul": [3, 4, 5, 6, 7, 8], "builder": [3, 4, 5, 6, 7, 8], "constant": 3, "100": [3, 4], "200": 3, "date": [3, 4], "2022": 3, "01": 3, "02": 3, "03": 3, "04": 3, "cap": 3, "10m": 3, "20m": 3, "market_cap": 3, "10000000": 3, "20000000": 3, "volum": 3, "1m": [3, 5], "2m": 3, "measu": 3, "usd": 3, "1000000": 3, "2000000": 3, "target_weight": 3, "75": [3, 4], "50": [3, 4], "30": 3, "00": 3, "veri": 3, "moder": 3, "size": 3, "market": [3, 4], "both": 3, "abl": 3, "number": [3, 4], "unrealist": 3, "trade_volum": 3, "max_cap_fract": 3, "06": [3, 4, 7], "max": 3, "6": 3, "max_trade_fract": 3, "20": 3, "daili": [3, 4], "8": 3, "smaller": 3, "won": 3, "collid": 3, "1e5": 3, "larger": 3, "far": 3, "off": 3, "target": 3, "1e7": 3, "08": [3, 4], "12": [3, 4], "option": [4, 5, 6, 7, 8], "plot": [4, 5, 6, 7, 8], "backend": [4, 5, 6, 7, 8], "plotli": [4, 5, 6, 7, 8], "yfinanc": 4, "yf": 4, "resample_index": 4, "download": 4, "ticker": 4, "spy": 4, "aapl": 4, "goog": 4, "msft": 4, "period": 4, "10y": 4, "time": [4, 5, 6, 7], "interv": 4, "1d": 4, "prepost": 4, "pre": 4, "post": 4, "hour": 4, "repair": 4, "obviou": 4, "error": 4, "100x": 4, "3": 4, "adj": 4, "close": [4, 5], "cumsum": 4, "usual": 4, "would": 4, "basi": 4, "rebal": 4, "everi": [4, 7], "week": 4, "month": 4, "There": 4, "deal": 4, "problem": 4, "cvxsimul": 4, "see": 4, "effect": 4, "hesit": 4, "most": 4, "flexibl": 4, "irregular": 4, "portfolio_resampl": 4, "rule": 4, "datafram": 4, "origin": 4, "monthli": 4, "2013": 4, "10": [4, 7], "000000e": 4, "11": 4, "9": 4, "893226e": 4, "05": 4, "831524e": 4, "831219e": 4, "13": 4, "884833e": 4, "884375e": 4, "14": [4, 7], "807532e": 4, "807026e": 4, "2023": 4, "7": 4, "825517e": 4, "808727e": 4, "832360e": 4, "815795e": 4, "07": 4, "673877e": 4, "656573e": 4, "722090e": 4, "704767e": 4, "09": [4, 7], "781098e": 4, "763541e": 4, "2519": 4, "print": [4, 8], "18370": 4, "629754": 4, "11275": 4, "283299": 4, "8458": 4, "031523": 4, "1825": 4, "371528": 4, "10761": 4, "214327": 4, "15582": 4, "431502": 4, "5827": 4, "130696": 4, "4594": 4, "345965": 4, "x": 4, "hard": 4, "between": 4, "trades_stock": 4, "iloc": 4, "els": 4, "forward": 4, "lead": 4, "150k": 4, "had": 4, "realloc": 4, "turnov": 4, "i": 4, "don": 4, "believ": 4, "bring": 4, "render": 4, "signal": 4, "spars": 4, "stai": 4, "numpi": [5, 6, 7], "stock_pric": [5, 6, 7, 8], "len": [5, 6, 8], "w": [5, 6], "rand": [5, 6], "sum": [5, 6], "one_over_n": 5, "diff": 5, "d": 5, "remain": 5, "financ": 5, "round": 5, "littl": 7, "exercis": 7, "goe": 7, "back": 7, "idea": 7, "stephen": 7, "boyd": 7, "should": 7, "even": 7, "like": 7, "Not": 7, "Of": 7, "cours": 7, "termin": 7, "go": 7, "bust": 7, "which": 7, "seem": 7, "loguru": 7, "logger": 7, "trading_cost": 7, "linearcostmodel": 7, "info": 7, "pai": 7, "fee": 7, "bp": 7, "factor": 7, "0000": 7, "trading_cost_model": 7, "assert": 7, "game": 7, "over": 7, "32m2023": 7, "40": 7, "32": 7, "247": 7, "0m": 7, "1minfo": 7, "36m__main__": 7, "36m": 7, "modul": 7, "36m1": 7, "1mload": 7, "259": 7, "36m5": 7, "1mbuild": 7, "260": 7, "36m8": 7, "support": 8, "recommend": 8, "qs": 8, "snapshot": 8, "titl": 8, "show": 8, "findfont": 8, "font": 8, "famili": 8, "arial": 8, "found": 8, "sharp": 8, "ratio": 8, "stat": 8, "pct_chang": 8, "dropna": 8, "6379901607052793": 8, "6389878459352839": 8, "line": 9, "code": 9}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"api": 0, "sphinx": 0, "depend": 0, "simul": 1, "modu": 1, "operandi": 1, "creat": 1, "builder": 1, "object": 1, "loop": 1, "through": 1, "time": 1, "analys": 1, "result": 1, "bypass": 1, "poetri": 1, "kernel": 1, "notebook": 2, "estim": 3, "capac": 3, "us": 3, "box": 3, "constraint": 3, "long": 4, "onli": 4, "1": 4, "n": [4, 5], "portfolio": [4, 5, 6], "rebalanc": 4, "resampl": 4, "an": 4, "exist": 4, "trade": [4, 7], "dai": 4, "predefin": 4, "grid": 4, "why": 4, "price": 4, "differ": 5, "monkei": [5, 6], "One": 5, "over": 5, "almost": 7, "pair": 7, "With": 8, "quantstat": 8, "ci": 9, "cd": 9, "report": 9, "loc": 9, "test": 9}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})