# Cultural Incentives and the Evolution of Artistic Honesty

Code which utilises replicator dynamics and agent-based methods to model the evolution of creative culture on a platform.

## Question to be investigated:
**How does virality-based reward reshape which artists survive**
Specifically, I modelled how a platform that preferentially amplifies certain content influences the long-run distribution of artist honesty (defined as the degree to which a creator produces socially responsible, rather than purely optimised work).

## Model overview:
**Agents:**

The population consists of instances of a creators class, each with their own honesty value \( h \in [0,1] \) where h = 0 represents fully optimised, engagement-maximising work and h = 1 represents fully socially/culturally responsible, authentic work. The initial distribution of the honesty is either continuous (agent-based simulation) or discrete (replicator dynamics simulation) over [0,1]

**Platform reward mechansim:**

the platform ranks content via a "content score". The score for player i is defined as:

\(c_{\text{i}} = \alpha_{\text{auth}} \, h_{\text{i}} + \alpha_{\text{opt}} \, (1-h_{\text{i}}) \)

I implemented a contest success function where creators receive attention proportional to their content score to some power (\gamma) relative to the whole population (sum of all the other scores each to this same power). It was defined as:

E_{\text{i}} = \frac{c_{\text{i}}^\gamma}{\sum_{\text{j}} c_{\text{j}}^\gamma}

A_{\text{i}} = B + k*(E_{\text{i}})

where B is the baseline guaranteed attention (if such an amount exists), and k is the total amount of attention to be allocated strategically.

**Evolutionary dynamics:**

I examined two evolutionary processes:

1. **replicator dynamics**, where the density proportion of creators with honesty type h in the population evolves proportionally to their relative payoff. The change in frequency of each type of honesty is governed by the equation:

\dot{f_{\text{i}}} = f_{\text{i}} \, (A_{\text{i}}) - \langle A \rangle)

where \langle A \rangle is the average reward (attention) for that frequency distribution. I used the Euler method to solve the differential equation numerically.

2. **Agent-based strategy updating**, where individual creators stochastically imitate the strategy of higher-payoff peers. At each time step, the reward for each player was calculated. Then they'd select a random peer and adopted their strategy with probability based on the Fermi rule:

P(s_{\text{i}} \rightarrow s_{\text{j}}) = \frac{1}{1-exp(-\beta(A_{\text{j}} - A_{\text{i}}))}

where \beta controls the intensity of selection.

Both processes select for creators whose payoff under the platform reward system is highest.

## Results:

Across both dynamics, mean honesty consistently declined from its initial value (provided the distribution had enough honesty values lower than the mean to work with in the case of the agent-based model). Under replicator dynamics, mean honesty monotonically decreased towards 0. The agent-based model exhibited a slower, noisier decline as a result of using the Fermi rule and never declined all the way to 0 (as we are only swapping honesties around, and there's no guarantee that we'll have enough 0-honesty players at any given point in time, with a game running for long enough, for us to observe a complete collapse). However, the trend was identical: the platform's virality-based reward structure reshapes the creator population toward increasingly engagement-maximising, culturally irresponsible creative content. 

[graphs]

**Proposition: 0 honesty is a dominant strategy solution for any content scoring function which decreases with honesty**

Proof: In order for 0 honesty to be a dominant strategy solution for player i, we need u_{\text{i}}(s_{\text{i}}, s_{\text{-i}}^\ast) \geq u_{\text{i}}(s_{\text{i}}^\ast, s_{\text{-i}}^\ast) \forall s^\ast \in S (!).

We adopt the convenient notation s = (s_{\text{i}}, s_{\text{-i}}), where s \in S (the set of all strategy vectors for this game), s_{\text{i}} is the strategy of player i and s_{\text{-i}} is the strategy vector minus player i's strategy.

note that u_{\text{i}}(s) = B + k(\frac{(s_{\text{i})^\gamma}{\sum_{\text{j}}(s_{\text{j}})^\gamma})

Lemma: \frac{x}{x+a} is monotonically increasing for x \geq 0.

Proof: \frac{x}{x+a}' = \frac{(x+a)(x)' - x(x+c)'}{(x+a)^2} = \frac{x+a-x}{(x+a)^2} = \frac{a}{(x+a)^2} \geq 0 when a \geq 0 

in order to decide if (!) holds we must compare 

B + k(\frac{(s_{\text{i})^\gamma}{s_{\text{i}}^\gamma + \sum_{\text{j \neq i}}((s_{\text{j}})\ast)^\gamma}) 

with 

B + k(\frac{((s_{\text{i})\ast)^\gamma}{\sum_{\text{j \neq i}}((s_{\text{j}})\ast)^\gamma}).

This is the same as comparing

\frac{(s_{\text{i})^\gamma}{s_{\text{i}}^\gamma + \sum_{\text{j \neq i}}((s_{\text{j}})\ast)^\gamma} (1)

with 

\frac{((s_{\text{i})\ast)^\gamma}{\sum_{\text{j}}((s_{\text{j}})\ast)^\gamma} (2)

these expressions are in the form \frac{x}{x+a}, a is the constant \sum_{\text{j \neq i}}((s_{\text{j}})\ast)^\gamma}. By our lemma, this means that the one for which the term representing x is bigger will be the larger one.

s_{\text{i}} is the strategy h = 0. By our scoring function, the attention from this strategy will be \alpha_{\text{opt}}. since the scoring function is monotonically decreasing with h, any other strategy for player i s_{\text{i}} must give a content score \leq \alpha_{\text{opt}} (since h \geq 0 for any other strategy). this means then, that whatever strategy we pick, since h \geq 0 for that strategy (meaning a content score at most as high as when h = 0), the (1) \geq (2) for any alternative strategy vector s \ast \in S. \square

The only way to avoid collapse to zero honesty in this model is if \alpha_{\text{auth}} \geq \alpha_{\text{opt}}. The conclusion we draw is:

Cultural responsibility is not evolutionarily stable unless the platform actively incentivises it.

this reflects classical results from mechanism design; that if a system rewards a property, equilibrium behaviour will favor that proprty, even when doing so will have negative externalities. 

## Limitations and Future Work

1. **Variable attention supply:**

This formulation of reward (attention) makes this a zero-sum game. The total amount of attention available to crreators at any given point in time is constant at nB + k where n is the number of creators. In reality, this would be elastic. platforms grow and shrink, genres trend, and user behaviour is time-dependent. A more realistic model should take this into account.

2. **Strategy mutation or innovation**

In the agent-based model, the mean honesty can never fall below the lowest honesty initially present in the population, because creators only imitate existing strategies. The prevents the system from reaching the analytical equilibrium of 0 honesty unless some creator has that strategy to begin with (and even then, there's always a non-zero chance that any creator with some honesty will trade it for a strategy with higher honesty. And if at any point in time there is no creator in the population with a particular honesty value, that honesty value will never emerge again in the population, meaning if all the 0 honesty creators trade their stratgies away, achieving the analytical equilibrium is impossible). A more complete evolutionary model would incorporate mutations, explorations, or continuous strategy drift. We might even implement the rank-shift model. This allows creators to discover lower honesties when doing so increases payoff.

3. **platform redesign and mechanism comparison**

Future models will test alternative platform mechanisms (eg mixed strategies, taxation on virality spikes, fairness weightings, etc.) to identify which mechanisms stabilise or recover honesty. This will turn the model into a mechanism design environment for cultural ecosystems. 

## Conclusions

this project demonstrates how incentive structures induce long-run behavioural collapse in creative ecosystems, connecting ideas from algorithmic game theory, evolutionary dynamics, and digital platform mechanism design. The results suggest that virality-based reward structures on platformmay systemically erode cultural responsibility unless honesty is explicitly built into the scoring function. 
