u were to build an architeture pipeline
dynamo db has demand profiles, multiple diff years, n diff sectors, each profile is stored via country, sector, year, location, each profile is hourly demand of year - 8760 vals
user uploads data profile, 8760 r gives partial data r some info abt his machines n stuff n pue n other necessary things which can be sued to estimate 8760 vals
1st is retrival, what n how will u retrive, is retrival really necessary

nxt is optimising, we've heuristic optimiser, fast n less optimal for browser, another Lp variant, takes time, but global optimal
heuristic is like - estimated time n cost, 
Lp is like exact time n cost,
when user is in session, we'll hit both endpoints, lp n optuna - heuristic endpoints at same time
we'll use results from optuna for that session, 
lp will still be running, 
we'll store the optuna result n session id - where??
after lp completion, we'll store lp rseults also, along with session id - where, like db r aws provides anything
at end of everyday - r another timeline - u decide we'll compare lp n heuristic results for every sector, country, location, year
if lp n optuna results deviate much, we'lll check hyp param of optuna for that group n try to make them is acceptable range of deviation - let's say deviation is normally distributed r we can construct distribution, we'll check deviation of vals above p75 r p90, if it's too large, we'll try to reduce via abov e method
is this pipeline good n we can change r add r remove antything
give ur answers in new md file, pointwise, in a flow, no need of code