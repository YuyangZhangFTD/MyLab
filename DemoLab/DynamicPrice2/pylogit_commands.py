import numpy as np
from numpy.matlib import repmat
import pandas as pd
import pylogit as pl
from collections import OrderedDict

data = pd.read_csv("D:/TrafficModeData.csv")
data["Group"] = repmat(np.array([x+1 for x in range(210)]).reshape(210,1),1,4).reshape(840,1)
data["ALT"] = repmat(np.array([1,2,3,4]).reshape(4,1),210,1)
print(data.head(8))

# # logit model-1: only generic coefficients
# spec = OrderedDict()
# variable_names = OrderedDict()
# Vars = ["TTME", "INVC", "INVT"]
# for var in Vars:
    # spec[var] = [[1,2,3,4]]
    # variable_names[var] = [var]
# model = pl.create_choice_model(data = data,
                               # alt_id_col="ALT",
                               # obs_id_col="Group",
                               # choice_col="MODE",
                               # specification=spec,
                               # model_type = "MNL",
                               # names = variable_names
                               # )
# model.fit_mle(np.zeros(3))
# model.print_summaries()

# # logit model-2: plus alternative-specific coefficients
# spec = OrderedDict()
# variable_names = OrderedDict()
# Vars = ["TTME", "INVC", "INVT"]
# for var in Vars:
    # spec[var] = [[1,2,3,4]]
    # variable_names[var] = [var]
# spec["intercept"] = [1,2,3]
# variable_names["intercept"] = ["ASC Air", "ASC Train", "ASC Bus"]
# spec["HINC"] = [1,2,3]
# variable_names["HINC"] = ["HINC for Air", "HINC for Train", "HINC for Bus"]
# model = pl.create_choice_model(data = data,
                               # alt_id_col="ALT",
                               # obs_id_col="Group",
                               # choice_col="MODE",
                               # specification=spec,
                               # model_type = "MNL",
                               # names = variable_names
                               # )
# model.fit_mle(np.zeros(9))
# model.print_summaries()


#logit model-3: plus alternative-specific coefficients
spec = OrderedDict()
variable_names = OrderedDict()
Vars = ["TTME", "INVC", "INVT"]
spec["intercept"] = [1,2,3]
variable_names["intercept"] = ["ASC Air", "ASC Train", "ASC Bus"]
for var in Vars:
    spec[var] = [[1,2,3,4]]
    variable_names[var] = [var]
spec["HINC"] = [4]
variable_names["HINC"] = ["HINC for Car"]
spec["TTME"] = [[1], [2,3,4]]
variable_names["TTME"] = ["TTME for Air", "TTME for Train/Bus/Car"]
model = pl.create_choice_model(data = data,
                               alt_id_col="ALT",
                               obs_id_col="Group",
                               choice_col="MODE",
                               specification=spec,
                               model_type = "MNL",
                               names = variable_names
                               )
model.fit_mle(np.zeros(8))
model.print_summaries()


# #retrive model parameters
# print("\n\nFollowings are attributes of the model object")
# print(dir(model))
# print("\nFollowings are coefficients")
# print(model.params.values)
# print("\nFollowings are pvalues")
# print(model.pvalues.values)

# predict
print("\n\nFollowing are choosing probabilities of all alternatives in the first 2 cases")
print(model.predict(data.iloc[0:8]))
print("\nFollowing are choosing probabilities of actually chosen alternatives in the first 2 cases")
print(model.predict(data.iloc[0:8],choice_col="MODE", return_long_probs=False))


# # nested logit model-1: using the specification of the logit model-2, and Fly-Ground nest 
# spec = OrderedDict()
# variable_names = OrderedDict()
# nest_membership = OrderedDict()
# nest_membership["Fly"] = [1]
# nest_membership["Ground"] = [2,3,4]
# Vars = ["TTME", "INVC", "INVT"]
# for var in Vars:
    # spec[var] = [[1,2,3,4]]
    # variable_names[var] = [var]
# spec["intercept"] = [1,2,3]
# variable_names["intercept"] = ["ASC Air", "ASC Train", "ASC Bus"]
# spec["HINC"] = [1,2,3]
# variable_names["HINC"] = ["HINC for Air", "HINC for Train", "HINC for Bus"]
# model = pl.create_choice_model(data = data,
                               # alt_id_col="ALT",
                               # obs_id_col="Group",
                               # choice_col="MODE",
                               # specification=spec,
                               # names = variable_names,
                               # model_type = "Nested Logit",
                               # nest_spec = nest_membership
                               # )
# model.fit_mle(np.zeros(11))
# model.print_summaries()

# # nested logit model-2: using the specification of the logit model-2, and Public-Private nest 
# spec = OrderedDict()
# variable_names = OrderedDict()
# nest_membership = OrderedDict()
# nest_membership["Private"] = [1,4]
# nest_membership["Public"] = [2,3]
# Vars = ["TTME", "INVC", "INVT"]
# for var in Vars:
    # spec[var] = [[1,2,3,4]]
    # variable_names[var] = [var]
# spec["intercept"] = [1,2,3]
# variable_names["intercept"] = ["ASC Air", "ASC Train", "ASC Bus"]
# spec["HINC"] = [1,2,3]
# variable_names["HINC"] = ["HINC for Air", "HINC for Train", "HINC for Bus"]
# model = pl.create_choice_model(data = data,
                               # alt_id_col="ALT",
                               # obs_id_col="Group",
                               # choice_col="MODE",
                               # specification=spec,
                               # names = variable_names,
                               # model_type = "Nested Logit",
                               # nest_spec = nest_membership
                               # )
# model.fit_mle(np.zeros(11))
# model.print_summaries()


# # mixed logit model: using the specification of the logit model-2
# spec = OrderedDict()
# variable_names = OrderedDict()
# Vars = ["TTME", "INVC", "INVT"]
# for var in Vars:
    # spec[var] = [[1,2,3,4]]
    # variable_names[var] = [var]
# spec["intercept"] = [1,2,3]
# variable_names["intercept"] = ["ASC Air", "ASC Train", "ASC Bus"]
# spec["HINC"] = [1,2,3]
# variable_names["HINC"] = ["HINC for Air", "HINC for Train", "HINC for Bus"]
# model = pl.create_choice_model(data = data,
                               # alt_id_col="ALT",
                               # obs_id_col="Group",
                               # choice_col="MODE",
                               # specification=spec,
                               # names = variable_names,
                               # model_type = "Mixed Logit",
                               # mixing_id_col="Group",
                               # mixing_vars = ["TTME", "INVC", "INVT"]
                               # )
# model.fit_mle(np.zeros(12), num_draws=1000)
# model.print_summaries()


