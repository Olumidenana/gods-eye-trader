"""The edge model: ICT supplies the hypotheses, the data judges them.

A trade is worth taking only if its expected result after costs is above zero. With a stop of 1R, a target of T R and costs of c R:

    EV = p * T - (1 - p) - c           p = the chance price reaches the target before the stop

On a random walk p = 1 / (1 + T), so EV = -c: every trade loses its costs, and no rule written on top of random prices changes that. An edge
exists only where ICT's ideas (the draw on liquidity, premium and discount, displacement, the array, the time) lift p above that baseline, and
only a measurement can say whether they do.

This package does the measuring:

  candidates  every sweep -> shift -> gap the engine can find (gates off), with the ICT features at the moment of the order and what price did next
              (the furthest it ran for the trade before the stop)
  model       a regularised logistic model of p for a grid of targets, pulled towards "no edge", each feature's direction fixed to what ICT teaches
  validate    walk-forward (train on the past, test on the next month), a cluster bootstrap interval and a permutation test: the verdict is
              EDGE only when the lower bound is above zero and chance is an unlikely explanation
  sizing      a fraction of the Kelly size, from the LOWER bound of p, with caps
  monitor     live results against the model's own predictions; when they drift apart the model switches itself off (it never re-tunes itself)

Nothing here changes the live engine. Run it on real candles first (python -m ict_trader.edge --help); wire a model into decisions only if its
verdict is EDGE on data it was not trained on.
"""
