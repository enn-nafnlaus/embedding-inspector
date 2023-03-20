import numpy as np
import torch
import scipy
from scipy.optimize import minimize
from collections import OrderedDict
import random, math, time, re, pickle, copy
import json

class EmbeddingGroupFinder:
  
  WEIGHTS_AS_TENSOR = False	# Submit weights as a tensor to scipy.optimize.minimize (though we seem to just get a np array out  :Þ)
  
  EMBEDDING_BASEDIR = "embeddings"
  EMB_LENGTH = 768
#  DO_SORT = False
  CALC_SHORTER_GROUPS = False
  DO_PRECACHE = False
#  SAVE_ALL = False

  MAX_SIMILAR_EMBS = 20
  MAX_HISTORY = 200000

#  USE_SORT_WEIGHTINGS = DO_SORT
  MULTIPLIER_LIST = [5,2.5,1.6,1.25,1]
  MIN_OPT_ITER = 8
  MAX_OPT_ITER = 80
  MAX_EMBS_PER_GROUP = 200
  OPT_SCALAR = 170
  USE_WEIGHT_PENALTY1 = False
  USE_WEIGHT_PENALTY2 = False	
  SPECIFIC_GROUP_ODDS = 0.5	# Odds that, when using SAME or NEAR embeddings, that it comes from a specific best_emb_group.
  SAME_EMB_ODDS = 0.5		# Single-pass odds from the chosen best_emb_group (1.0 = guaranteed to repeat the previous-best embeddings)
  SAME2_EMB_ODDS = 0.7		# Same thing, but from any best_emb_group.
  NEAR_EMB_ODDS = 0.9		# Repeated odds (1.0 = will repeat a neighbor of a previous-best embedding over and over again ad infinitum - never use 1.0!)
  OPTIONAL_EMB_ODDS = 0.75	# Since this is also a while-loop breakout odds, never use 1.0!
  METHOD = "Powell"
  
  START_OPTIMIZATION_EVALUATION = 60 * 20	# Wait this many seconds before starting to evaluate our optimization methodology's efficacy
  OPTIMIZATION_PRINT_FREQUENCY = 60 * 60 * 24 * 365	# Seconds
  SUCCESSIVE_OPTIMIZATION_RUNS = 30
  SAVE_FREQUENCY = 60 * 30			# Seconds

  RUNTIME = 60 * 60 * 24 * 365.24	# Seconds
  MAXITER = 10000000			# Count
  
  BEST_EMB_GROUPS = 1000
  NEAREST_EMB_COUNT = 50
  PRINT_DURING_CALC = False
  
#  DEST_FILENAME = None
  REQUIRED_TOKENS_TEXT = []
  OPTIONAL_TOKENS_TEXT = []

  LOAD_ON_CPU = not torch.cuda.is_available()	# Load torch models on the CPU, in case the GPU is unavailable
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  OPTIMIZATION_CHOICES = {
#    "USE_SORT_WEIGHTINGS": [True, False],
#    "MULTIPLIER_LIST": [[300,45,12,4,2.2,1.5,1.2,1], [100,20,7,3,2,1.5,1.2,1], [50,15,6,3,2,1.5,1.2,1], [20,7,3,2,1.5,1.2,1], [7,2.5,1.5,1.2,1], [3,1.5,1.2,1], [1]],
#    "MULTIPLIER_LIST": [[50,15,6,3,2,1.5,1.2,1], [20,7,3,2,1.5,1.2,1], [7,2.5,1.5,1.2,1], [3,1.5,1.2,1], [1]],
    "MULTIPLIER_LIST": [[6,2.5,1.5,1], [5,2.2,1.4,1], [4,2,1.3,1], [3,2,1.5,1.2,1], [2.5,1.4,1]],
#    "MULTIPLIER_LIST": [ [1] ],
    "MIN_OPT_ITER": [2,3,4,5,8],
#    "OPT_SCALAR": [150,180,220,300],
#    "USE_WEIGHT_PENALTY1": [True, False],	# Nánast eins...
#    "USE_WEIGHT_PENALTY2": [True, False],
#    "SPECIFIC_GROUP_ODDS": [0.0, 0.25, 0.5, 0.75, 1.0],
#    "SAME_EMB_ODDS": [0.0, 0.5, 0.9, 1.0],
#    "SAME2_EMB_ODDS": [0.55, 0.7, 0.85],		# Never use 1.0!
#    "NEAR_EMB_ODDS": [0.7, 0.9, 0.95],			# Never use 1.0!
#    "OPTIONAL_EMB_ODDS": [0.65, 0.75, 0.85],		# Never use 1.0!
#    "METHOD": ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]
  }


  do_interrupt = False
  do_breakout = True
  do_skip = False
  do_printout = False
  textbox = ""
  mix_inputs = []
  mix_sliders = []

  def get_outputs(self, *args):
    global MAX_NUM_MIX
    
    if self.do_breakout:
      return args
    else:
      append_count = MAX_NUM_MIX - len(self.mix_inputs)
      self.textbox = self.textbox[:self.textbox.rindex("\n") + 1]
      return tuple([self.textbox] + self.mix_inputs + [None]*append_count + self.mix_sliders + [0]*append_count)


  def set_required(self, arg):
    self.REQUIRED_TOKENS_TEXT = [i.split(",") for i in arg.split(";")]

    
  def set_optional(self, arg):
    self.OPTIONAL_TOKENS_TEXT = arg.split(";")

    
  def set_count(self, arg):
    self.MAX_SIMILAR_EMBS = arg

    
  def set_precache(self, arg):
    self.DO_PRECACHE = arg

    
  def interrupt(self):
    self.do_interrupt = True
    self.textbox.change( + "\n\nInterrupt requested")


  def breakout(self):
    self.do_breakout = True
    self.textbox.change( + "\n\nBreakout requested")


  def save_near_emb_cache(self):
    with open(self.EMBEDDING_FILENAME + ".near", "wb") as outfile:
      pickle.dump(self.near_emb_cache, outfile)
  

  def load_near_emb_cache(self):
    self.textbox += "Loading cached near embeddings (if present)...\n"
    try:
      with open(self.EMBEDDING_FILENAME + ".near", "rb") as infile:
        self.near_emb_cache = pickle.load(infile)
    except FileNotFoundError:
      self.textbox += "  ...Not found.\n"
      pass


#  def save_state(self):
#    save_keys = {i:self.__dict__[i] for i in self.__dict__.keys()}
#    for i in ["EMB_LENGTH", "all_embs", "orig_all_embs", "sorted_to_orig", "orig_to_sorted", "emb_weights", "emb_indices", "OPTIMIZATION_CHOICES", "PRINT_DURING_CALC", "START_OPTIMIZATION_EVALUATION", "OPTIMIZATION_PRINT_FREQUENCY", "SUCCESSIVE_OPTIMIZATION_RUNS", "SAVE_FREQUENCY", "USE_SORT_WEIGHTINGS", "MULTIPLIER_LIST", "MIN_OPT_ITER", "MAX_OPT_ITER", "OPT_SCALAR", "", "USE_WEIGHT_PENALTY1", "USE_WEIGHT_PENALTY2", "SPECIFIC_GROUP_ODDS", "SAME_EMB_ODDS", "SAME2_EMB_ODDS", "NEAR_EMB_ODDS", "OPTIONAL_EMB_ODDS", "METHOD"]:
#      try:
#        del save_keys[i]
#      except KeyError:
#        pass
#
#    with open("saved_state.minimize", "wb") as outfile:
#      pickle.dump(save_keys, outfile)
      

#  do_resume = False
#  
#  def resume(self):
#    with open("saved_state.minimize", "rb") as infile:
#      tmp_dict = pickle.load(infile)
#    self.__dict__.update(tmp_dict) 
#    try:
#      self.load_embeddings(self.EMBEDDING_FILENAME)
#      self.do_resume = True
#    except RuntimeError:
#      self.do_resume = False
#    return self.do_resume


#  def save_best(self, best_emb_groups_list, iterations=1, score=None, checkpoint='v2-1_768-ema-pruned', checkpoint_hash='4bdfc29c', string_to_token=265):
#    width = 0
#    for i in best_emb_groups_list:
#      if i is not None and len(i) > width:
#        width = len(i)
#
#    if width > 0:
#      solve_groups_tensor = torch.stack(best_emb_groups_list)
#    else:
#      return
#      
#    match = re.match("^(.*/)(.+?)\.([^\./]+)$", self.DEST_FILENAME)
#    if not match:
#      match = re.match("^()(.+?)\.([^\./]+)$", self.DEST_FILENAME)
#    if not match:
#      self.textbox += "\n"
#      self.textbox += "Invalid filename:" + str(self.DEST_FILENAME) + "\n"
#      self.textbox += "\n"
#      return
#    path = match.group(1)
#    base = match.group(2)
#    extension = match.group(3)
#    if score is None:
#      filename = self.DEST_FILENAME
#    else:
#      filename = f"{path}{base}_%08d_%03d.bin" % (iterations, round(score * 1000))
#    if extension == "bin":
#      torch.save({f"<{base}>": best_emb_groups_list[0]}, filename)
#    if extension == "pt":
#      torch.save({'string_to_token': {'*': string_to_token}, 'string_to_param': {'*': best_emb_groups_list}, 'name': base, 'step': iterations, 'sd_checkpoint': checkpoint_hash, 'sd_checkpoint_name': checkpoint}, filename)


  emb_indices = None
  emb_weights = None
  
  def pick_emb(self):
#    if self.USE_SORT_WEIGHTINGS == True:
    return np.random.choice(self.emb_indices, size=1, p=self.emb_weights)[0]
#    else:
#      return random.randint(0, len(self.all_embs) - 1)
  
  
  torch_cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
  
  
  def emb_difference(self, a, b):
    return 1 - float(self.torch_cos(a, b))


  def magnitude(self, weights):
    return torch.linalg.norm(weights)

  
  def find_near_embs(self, a):
    list = [ (self.emb_difference(self.orig_all_embs[emb_id], a), emb_id) for emb_id in range(len(self.orig_all_embs)) ]
    list.sort()
    return np.array([idx for score, idx in list[:self.NEAREST_EMB_COUNT]], dtype=np.int32)

  
  def get_group_vec(self, weights, emb_group):
    w = torch.unsqueeze(torch.Tensor(weights).to(self.DEVICE), dim=-1)
    tensor_list = w * emb_group
    return torch.sum(tensor_list, dim=0)

  
  def difference_from_target_emb(self, weights):
    w = torch.unsqueeze(torch.Tensor(weights).to(self.DEVICE), dim=-1)
    tensor_list = w * self.new_emb_group_deref
    group_vec = torch.sum(tensor_list, dim=0)
    return (1 - float(self.torch_cos(group_vec, self.target_emb)))
  

  def time_str(self):
    return "\r %0.1f%% " % min(100, (100 * (time.time() - self.start_time) / (self.end_time - self.start_time)))


  def print_best(self, best_emb_groups_list, partial=False):
    if best_emb_groups_list is None:
      return
    
    for best_emb_groups, iterations in best_emb_groups_list:
      if best_emb_groups is None:
        continue
      self.textbox += f"\n\n%0.1f :: Iteration: {iterations}\n" % (time.time() - self.start_time)
      for subset_id in [len(best_emb_groups) - 1]:
        subset = best_emb_groups[subset_id]
        if len(subset) == 0:
          continue
        score, group_vec, emb_group, emb_set = subset[0]
        self.textbox += f"Score: {score}:\n\n"
        self.mix_inputs = []
        self.mix_sliders = []
        for weight, emb_id, mapped_emb_id in sorted(emb_group):
          self.mix_inputs.append(emb_id_to_name(int(mapped_emb_id), self.tokenizer))
          self.mix_sliders.append(weight)
          self.textbox += f" {weight} * {mapped_emb_id}: {emb_id_to_name(int(mapped_emb_id), self.tokenizer)}\n\n"
        if partial == True:
          return self.textbox + "---------------------------------\n\n"
        s = sorted(self.target_emb)
        rel_diff = s[int(0.75 * len(self.target_emb))] - s[int(0.25 * len(self.target_emb) )]
        self.textbox += f"this_emb: {group_vec.detach().numpy()}\n"
        self.textbox += f"target_emb: {self.target_emb.detach().numpy()}\n"
        self.textbox += f"difference: {((group_vec - self.target_emb) / rel_diff).detach().numpy()}\n\n\n"
      self.textbox += "-------------------------------------------------------------------------------------------------------\n\n"
  

  target_emb = None
  sorted_to_orig = {}
  orig_to_sorted = {}
  TARGET_FILENAME = None

  def load_target(self, filename, textbox):
    if filename == "":
      return textbox
    self.TARGET_FILENAME = filename
    textbox += f"\n\nLoading target '{filename}'... "
    try:
      if egf.LOAD_ON_CPU:
        models = torch.load(self.TARGET_FILENAME, map_location=torch.device('cpu'))
      else:
        models = torch.load(self.TARGET_FILENAME)
    except FileNotFoundError:
      textbox += "ERROR, file not found.\n"
      return textbox
    try:			# If it's a PT file, it may look something like the below.
      models = models['string_to_param']['*']
      if egf.LOAD_ON_CPU:
        models = [ i.float() for i in models ]
    except (KeyError, TypeError):	# Well, try whatever key exists... a .bin example looks like this:
      models = models[list(models.keys())[0]]
      if egf.LOAD_ON_CPU:
        models = models.float()	# Can't use half-precision on CPU.
      models = [ models ]        
    self.set_targets(models)
    textbox += "loaded.\n"
    return textbox

  
  def set_targets(self, t):
    self.target_embs = t


  def set_target(self, t):
    self.target_emb = t
    self.target_emb_mag = self.magnitude(self.target_emb)
    
  
#  def set_dest_filename(self, f):
#    self.DEST_FILENAME = f


  EMBEDDING_FILENAME = ""
#  
#  def load_embeddings(self, filename):
#    self.EMBEDDING_FILENAME = filename
#    if self.LOAD_ON_CPU:
#      e = torch.load(filename, map_location=torch.device('cpu')).float()
#    else:
#      e = torch.load(filename)
#    self.set_embeddings(e)


  def encode_token(self, t):
    t = t.lower()
    try:
      if t[-1] == "*":
        return self.orig_to_sorted[text_to_emb_ids([t[:-1]])]
      else:
        return self.orig_to_sorted[text_to_emb_ids(t + "</w>", self.tokenizer)]
    except KeyError:
      try:
        if t[-1] == "*":
          return self.orig_to_sorted[text_to_emb_ids(t[:-1] + "</w>", self.tokenizer)]
        else:
          return self.orig_to_sorted[text_to_emb_ids(t, self.tokenizer)]
      except KeyError:
        self.textbox += f"ERROR: could not find token '{t}' or '{t + '</w>'}'\n\n"
        self.textbox += "Suggested possibilities:\n"
        found = False
        sorted_keys = sorted(token_encoder.keys())
        for i in range(len(t)-1, -1, -1):
          t2 = t[:i]
          for j in sorted_keys:
            if j[:len(t2)] == t2:
              self.textbox += " " + str(j) + "\n"
              found = True
          if found:
            self.textbox += "\n"
            break
        if not found:
          self.textbox += "  <None>\n"
        raise ValueError(f"Token {t} does not exist")


  all_embs = None
  orig_all_embs = None
  required_tokens = []
  optional_tokens = []
  required_tokens_flattened = None

  def set_embeddings(self, e):
    if self.target_embs is None:
      self.textbox += "ERROR: Target should be set before embeddings are provided.\n"
      raise RuntimeError(f"Target should be set before embeddings are provided")

    self.EMB_LENGTH = len(e[0])
    self.all_embs = e
    self.orig_all_embs = e

    embedding_hash = int(torch.sum(e))	# For a more collision-secure hash, use hash(e.numpy.tostring()), though it's slower.
    self.EMBEDDING_FILENAME = self.EMBEDDING_BASEDIR + "%08X.hash" % embedding_hash

    self.load_near_emb_cache()
    


  def load_required_tokens(self):
    self.required_tokens = [ [ self.encode_token(i) for i in j ] for j in self.REQUIRED_TOKENS_TEXT ]
    self.required_tokens_flattened = [item for sublist in self.required_tokens for item in sublist]

    
  def load_optional_tokens(self):
    self.optional_tokens = [ self.encode_token(i) for i in self.OPTIONAL_TOKENS_TEXT ]


  cur_optimization_choices = {}
  optimization_records = {}

  def select_optimization_methodology(self):
    for key in self.OPTIMIZATION_CHOICES:
      choice = random.choice(self.OPTIMIZATION_CHOICES[key])
      setattr(self, key, choice)
      self.cur_optimization_choices[key] = choice

#    self.USE_SORT_WEIGHTINGS = random.choice([False] + [True] * 3)
    self.SAME_EMB_ODDS = random.random() ** 0.1
    self.SAME2_EMB_ODDS = (0.5 + 0.5 * random.random())  ** (20 / self.MAX_SIMILAR_EMBS)
    self.NEAR_EMB_ODDS = random.random() ** (0.5 / self.MAX_SIMILAR_EMBS)
    self.OPTIONAL_EMB_ODDS = (0.65 + 0.35 * random.random())  ** (10 / self.MAX_SIMILAR_EMBS)
    self.SPECIFIC_GROUP_ODDS = random.random()
    

  def new_optimize_rec(self, key, score_improvement, time_since_last):
    if key not in self.optimization_records:
      self.optimization_records[key] = [0.0, 0, 0.0]
    rec = self.optimization_records[key]
    rec[0] *= rec[2]
    rec[0] += score_improvement		# De-normalize the score and add the new one
    rec[2] += time_since_last		# Increment time
    rec[0] /= rec[2]			# Re-normalize the score
    rec[1] += 1				# Add to the iter count


  last_printed_optimization_time = 0
#  last_save_state_time = 0
  last_save_near_emb_time = 0
  last_avg_score = 0

  def evaluate_optimization_methodology(self, last_time, score_improvement):
    cur_time = time.time()
    score_improvement -= 1
    time_since_start = cur_time - self.start_time
    time_since_last = cur_time - last_time
    time_since_print = cur_time - self.last_printed_optimization_time
    if time_since_start < self.START_OPTIMIZATION_EVALUATION:
      return
    keylist = list(self.cur_optimization_choices.keys())
    for key1_id in range(len(keylist)):
      key1 = keylist[key1_id]
      val1 = self.cur_optimization_choices[key1]
      self.new_optimize_rec(f"{key1}:{val1}", score_improvement, time_since_last)
      for key2_id in range(key1_id + 1, len(keylist)):
        key2 = keylist[key2_id]
        val2 = self.cur_optimization_choices[key2]
        key = f"{key1}:{val1} & {key2}:{val2}"
        key_b = f"{key2}:{val2} & {key1}:{val1}"
        if key_b in self.optimization_records:
          key = key_b
        self.new_optimize_rec(key, score_improvement, time_since_last)
    if time_since_print > self.OPTIMIZATION_PRINT_FREQUENCY:
      best_optimization_records = [ (self.optimization_records[key], key) for key in self.optimization_records ]
      best_optimization_records.sort()
      for rec, key in best_optimization_records:
        self.textbox += "*/t=%0.7f (%s), #=%d, t=%0.1f\n" % (rec[0], key, rec[1], rec[2])
      self.textbox += "\n"
      self.last_printed_optimization_time = cur_time
        
  # ------------------
    
  start_time = None
  end_time = None
  
  best_emb_groups = None
  near_emb_cache = None
  iterations = None
  last_best_score = None
  history = None
  emb_id = 0
  solve_ret = []
  solve_groups = []
  total_iterations = 0
  tokenizer = None
  
  def solve(self):
    global SHOW_NUM_MIX
    
    self.MAX_SIMILAR_EMBS = SHOW_NUM_MIX
    self.do_breakout = False
    self.do_interrupt = False
    self.textbox = ""

    self.tokenizer, internal_embs, loaded_embs = get_data()
    self.set_embeddings(self.internal_embs)
    internal_embs = loaded_embs = None
    
#    if not self.do_resume:
    self.solve_ret = []
    self.solve_groups = []
    self.total_iterations = 0
    self.do_breakout = False
    self.emb_id = 0

    while self.emb_id < len(self.target_embs):
      t = self.target_embs[self.emb_id]
      self.textbox += f"\n === Embedding # {self.emb_id+1} / {len(self.target_embs)} === \n\n"
      
#      if not self.do_resume:
      self.set_target(t)
      self.best_emb_groups = None
      self.iterations = None
      self.last_best_score = None
      self.history = None
      self.start_time = None
      self.end_time = None

      self.all_embs = copy.deepcopy(self.orig_all_embs)
      tmp_emb_mapping = torch.Tensor( [i for i in range(len(self.orig_all_embs))] ).int().to(self.DEVICE)
  
      removed = len(self.all_embs)
      empty_mask = torch.all(self.all_embs == 0, axis=1)
  
      self.all_embs = self.all_embs[~empty_mask]
      tmp_emb_mapping = tmp_emb_mapping[~empty_mask]
      removed -= len(self.all_embs)

      if removed != 0:
        self.textbox += f"Removed {removed} empty embeddings.\n"

      weighted_embs = [ (self.emb_difference(self.all_embs[emb_id2], self.target_emb), tmp_emb_mapping[emb_id2], self.all_embs[emb_id2]) for emb_id2 in range(len(self.all_embs)) ]
#      if self.DO_SORT == True:
      self.textbox += "Sorting embeddings...\n"
      weighted_embs.sort(key=lambda x: x[0])

      self.textbox += "Finishing loading...\n"
      self.all_embs = [ emb for diff, emb_id2, emb in weighted_embs ]
      self.sorted_to_orig = { int(id): int(weighted_embs[id][1]) for id in range(len(weighted_embs)) }
      self.orig_to_sorted = { int(y): int(x) for x, y in self.sorted_to_orig.items() }
      self.emb_weights = np.array([ 1.0 / (diff + 1e-20) for diff, emb_id2, emb in weighted_embs ])
      self.emb_weights /= np.sum(self.emb_weights)

      try:
        self.load_required_tokens()
        self.load_optional_tokens()
      except ValueError:
        return

      self.emb_indices = np.array(range(len(self.all_embs)))

      best_emb_groups, iterations = self.solve_one()

      if len(best_emb_groups[-1]) > 0:
        beg_tuple = list(best_emb_groups[-1][0])
        l = []
        for i in range(len(beg_tuple[2])):
          l.append( (beg_tuple[2][i][0].detach().numpy(), beg_tuple[2][i][1], beg_tuple[2][i][2]) )
        beg_tuple[2] = l
        best_emb_groups = [[tuple(beg_tuple)]]
  
        self.solve_ret.append( (best_emb_groups, iterations) )
        self.solve_groups.append(best_emb_groups[-1][0][1].detach())
      else:
        self.solve_ret.append( (None, 0) )
        self.solve_groups.append(None)

      self.total_iterations += iterations
         
      self.emb_id += 1

      if self.do_breakout:
        break

#    self.save_best(self.solve_groups, self.total_iterations)
    self.print_best(self.solve_ret, self.iterations)
    
    self.do_breakout = True

    return self.solve_ret
      
  
  def solve_one(self):
    if self.best_emb_groups is None:
      self.best_emb_groups = [ [] for i in range(self.MAX_SIMILAR_EMBS) ]
      self.iterations = 0
      self.last_best_score = 1e50
    
    if self.near_emb_cache is None:
      self.near_emb_cache = [ [] for i in range(len(self.orig_all_embs)) ]
    
    if self.history is None:
      self.history = OrderedDict()
    
    self.do_interrupt = False

    self.start_time = time.time()
    self.end_time = self.start_time + self.RUNTIME
    
    to_precache = []
    if self.DO_PRECACHE:
      for i in range(len(self.near_emb_cache)):
        if len(self.near_emb_cache[i]) < self.NEAREST_EMB_COUNT:
          to_precache.append(i)
      if len(to_precache) > 0:
        self.textbox += "Precaching neighbors...\n"
    
    self.textbox += "\n"
    
    while time.time() < self.end_time and self.iterations < self.MAXITER and self.do_breakout == False and self.do_interrupt == False:
      self.textbox += self.time_str() + "                                           \r"
      last_time = time.time()
      
      if self.do_printout == True:
        self.do_printout = False
        self.print_best([(self.best_emb_groups, self.iterations)])
#        self.save_state()
        self.save_near_emb_cache()

      if len(to_precache) > 0:
        self.textbox += self.time_str() + f"<Finding neighbors to {to_precache[0]} -> {emb_id_to_name(to_precache[0, self.tokenizer])}>                        \r"
        self.near_emb_cache[to_precache[0]] = self.find_near_embs(self.orig_all_embs[to_precache[0]])
        to_precache = to_precache[1:]
        if len(to_precache) == 0:
          self.textbox += self.time_str() + f"<Precaching complete!>                        \r"
          self.save_near_emb_cache()
          self.textbox += "\n"
        elif last_time - self.last_save_near_emb_time > self.SAVE_FREQUENCY:
          self.textbox += self.time_str() + f"<Saving emb cache>                        \r"
          self.save_near_emb_cache()
          self.last_save_near_emb_time = last_time
        continue
      
      if self.iterations % self.SUCCESSIVE_OPTIMIZATION_RUNS == 0:
        self.select_optimization_methodology()
      
#      time_since_save_state = last_time - self.last_save_state_time
#      if time_since_save_state > self.SAVE_FREQUENCY and self.iterations != 0:
#        self.textbox += self.time_str() + f"<Saving state>                        \r"
#        self.save_state()
#        self.last_save_state_time = last_time
      
      cur_embs_per_group = min(self.MAX_EMBS_PER_GROUP, round(self.MAX_SIMILAR_EMBS * self.MULTIPLIER_LIST[0]))
      embs_thusfar = 0

      self.textbox += self.time_str() + f"<Creating embedding group>                        \r"
      new_emb_group = np.full((cur_embs_per_group,), -1, dtype=np.int32)
      for requirement_options in self.required_tokens:
        new_emb_group[embs_thusfar] = random.choice(requirement_options)
        embs_thusfar += 1
      while random.random() < self.OPTIONAL_EMB_ODDS and embs_thusfar < cur_embs_per_group:
        idx = random.choice(self.optional_tokens)
        if idx not in new_emb_group:
          new_emb_group[embs_thusfar] = idx
          embs_thusfar += 1
      if len(self.best_emb_groups[-1]) > 0:
        chosen_emb_group = random.choice(self.best_emb_groups[-1])		# Pick a previous-best embedding group
        for i in range(self.MAX_SIMILAR_EMBS):
          if embs_thusfar >= cur_embs_per_group:
            break
          if random.random() <= self.SAME_EMB_ODDS and chosen_emb_group[2][i][1] not in new_emb_group:
            new_emb_group[embs_thusfar] = chosen_emb_group[2][i][1]
            embs_thusfar += 1
        id = 0
        while random.random() < self.SAME2_EMB_ODDS and embs_thusfar < cur_embs_per_group:
          idx = random.choice(self.best_emb_groups[-1])[2][id][1]
          id = (id + 1) % self.MAX_SIMILAR_EMBS
          if idx not in new_emb_group:
            new_emb_group[embs_thusfar] = idx
            embs_thusfar += 1
        id = 0
        while random.random() < self.NEAR_EMB_ODDS:
          if random.random() < self.SPECIFIC_GROUP_ODDS:
            idx = chosen_emb_group[2][id][2]
          else:
            idx = random.choice(self.best_emb_groups[-1])[2][id][2]
          id = (id + 1) % self.MAX_SIMILAR_EMBS
          if embs_thusfar >= cur_embs_per_group:
            break
          if len(self.near_emb_cache[idx]) < self.NEAREST_EMB_COUNT:
            self.textbox += self.time_str() + f"<Finding neighbors to {idx} -> {emb_id_to_name(idx, self.tokenizer)}>                        \r"
            self.near_emb_cache[idx] = self.find_near_embs(self.orig_all_embs[idx])
            if last_time - self.last_save_near_emb_time > self.SAVE_FREQUENCY:
              self.textbox += self.time_str() + f"<Saving emb cache>                        \r"
              self.save_near_emb_cache()
              self.last_save_near_emb_time = last_time
            self.textbox += self.time_str() + f"<Creating embedding group>                        \r"
          for k in range(100):				# Up to 100 tries to find something that's not already there.
            mapped_choice = random.choice(self.near_emb_cache[idx])
            try:
              choice = self.orig_to_sorted[mapped_choice]
            except KeyError:
              continue
            if choice in new_emb_group:
              continue
            new_emb_group[embs_thusfar] = choice
            embs_thusfar += 1
            break
  
      # Fill the rest up with random embeddings.
      while embs_thusfar < cur_embs_per_group:
        for j in range(10000):
          choice = self.pick_emb()
          if choice in new_emb_group:
            continue
          new_emb_group[embs_thusfar] = choice
          embs_thusfar += 1
          break
        if j == 10000:	# Shouldn't happen...
          break

      new_emb_tuple = tuple(new_emb_group)

      # Don't repeat.
      try:
        self.history.pop(tuple(new_emb_tuple))
        self.history[new_emb_tuple] = None	# Move it to the back
        continue
      except KeyError:
        pass
      self.history[new_emb_tuple] = None
      if len(self.history) > self.MAX_HISTORY:
        self.history.popitem(last=False)
    
      self.textbox += self.time_str() + f"<Dereferencing embedding group>                        \r"
      # Dereference the embedding indices for speed in the optimization loop.
      self.new_emb_group_deref = torch.zeros([cur_embs_per_group, self.EMB_LENGTH])
      if self.WEIGHTS_AS_TENSOR:
        new_weights = torch.Tensor([(1 / cur_embs_per_group) ** 0.5 * random.choice([-1,1])] * cur_embs_per_group).to(self.DEVICE)
      else:
        new_weights = np.array([(1 / cur_embs_per_group) ** 0.5 * random.choice([-1,1])] * cur_embs_per_group)
    
      for i in range(cur_embs_per_group):
        self.new_emb_group_deref[i] = self.all_embs[new_emb_tuple[i]]
    

      # Progressively shrink down the list of embeddings under consideration.
      emb_count_list = [round(self.MAX_SIMILAR_EMBS * i) for i in self.MULTIPLIER_LIST] + list(range(self.MAX_SIMILAR_EMBS - 1, 0, -1))
      for emb_count_id in range(len(emb_count_list)):
        cur_embs_per_group = emb_count_list[emb_count_id]

        if self.CALC_SHORTER_GROUPS:
          if cur_embs_per_group == 1:
            break
        else:
          if cur_embs_per_group < self.MAX_SIMILAR_EMBS:
            break
    
        # Calculate the optimal weights for this set of self.all_embs.
        self.textbox += self.time_str() + f"<Running optimization loop, {self.METHOD}, {len(new_weights)} entries>                             \r"
        result = scipy.optimize.minimize(self.difference_from_target_emb, new_weights, tol=5e-3, method=self.METHOD, options={"maxiter": max(self.MIN_OPT_ITER, min(self.MAX_OPT_ITER, round(self.OPT_SCALAR / (cur_embs_per_group ** 0.5))))})
        new_weights = result.x
        score = result.fun
        self.textbox += self.time_str() + f"<Storing best results>                             \r"
        
        # Pick the most relevant embeddings in the group, eliminate the rest, then recalculate weights for the smaller group. Start with sorting by weights.
        next_embs_per_group = emb_count_list[emb_count_id + 1]
        composite = [ (new_weights[i], new_emb_tuple[i], self.sorted_to_orig[new_emb_tuple[i]]) for i in range(len(new_emb_tuple)) ]
        composite.sort(key=lambda x: abs(x[0]) + (1e50 if x[1] in self.required_tokens_flattened else 0))

        if cur_embs_per_group > self.MAX_SIMILAR_EMBS:
          composite_sort = composite[:]
          failed_narrowdown = True
          do_show = OrderedDict()
          for i in composite:
            do_show[i[1]] = True
          for i in range(100):
            subset_tuple = tuple([emb_group for weights, emb_group, mapped_emb_group in composite_sort[-next_embs_per_group:]])
            try:
              self.history.pop(subset_tuple)
              self.history[subset_tuple] = None		# Move it to the back
              bin_i = [not int(bit) for bit in bin(i + 2)[2:]]	#"2:"cuts the "0b" off the front.  +2 so the binary string ending starts with ...10
              if len(bin_i) > next_embs_per_group:
                break
              bin_i = [True] * (next_embs_per_group - len(bin_i) + 1) + bin_i
              if len(bin_i) > len(composite):
                break
              for j in range(len(bin_i)):
                do_show[composite[j][1]] = bin_i[j]
              composite_sort.sort(key=lambda x: abs(x[0]) + (1e50 if x[1] in self.required_tokens_flattened else 0) - 1e10 if do_show[x[1]] == 0 else 0)
              continue
            except KeyError:			# Not in history - good!
              subset = composite_sort[-next_embs_per_group:]
              failed_narrowdown = False
              break
          
          if failed_narrowdown:
            break
          
        else:
          subset = composite[-next_embs_per_group:]

        self.history[subset_tuple] = None
        if len(self.history) > self.MAX_HISTORY:
          self.history.popitem(last=False)
        
        # Before any elimination, though, store the best results, for each possible number of component embeddings.
        if cur_embs_per_group <= self.MAX_SIMILAR_EMBS:
          group_vec = self.get_group_vec(new_weights, self.new_emb_group_deref)
          group_vec_mag = self.magnitude(group_vec)
          mag_diff = self.target_emb_mag / group_vec_mag
          found = False
          new_entry_set = set(new_emb_tuple)
          new_entry = (score, group_vec * mag_diff, [ (weights * mag_diff, emb_group, mapped_emb_group) for weights, emb_group, mapped_emb_group in composite], new_entry_set)
          for i in range(len(self.best_emb_groups[cur_embs_per_group - 1])):
            if set(self.best_emb_groups[cur_embs_per_group - 1][i][3]) == new_entry_set:	# If this group already exists, but with different weightings, replace it.
              self.best_emb_groups[cur_embs_per_group - 1][i] = new_entry
              found = True
              break
          if not found:
            self.best_emb_groups[cur_embs_per_group - 1].append(new_entry)			# Does not exist; add it in.
          self.best_emb_groups[cur_embs_per_group - 1].sort(key=lambda x: x[0])			# Sort by score
          self.best_emb_groups[cur_embs_per_group - 1] = self.best_emb_groups[cur_embs_per_group - 1][:self.BEST_EMB_GROUPS]	# Keep only the best

        self.textbox += self.time_str() + f"<Preparing for next test>                             \r"
    
        # Prepare the next group of embeddings to test.
        composite = subset
#        composite = composite[-next_embs_per_group:]
        new_emb_group = np.array([emb_group for weights, emb_group, mapped_emb_group in composite], dtype=np.int32)
        new_emb_tuple = tuple(new_emb_group)
        self.new_emb_group_deref = torch.zeros([len(new_emb_group), self.EMB_LENGTH])
        if self.WEIGHTS_AS_TENSOR:
          new_weights = torch.Tensor([ weights for weights, emb_group, mapped_emb_group in composite]).to(self.DEVICE)
        else:
          new_weights = np.array([ weights for weights, emb_group, mapped_emb_group in composite])
    
        for i in range(next_embs_per_group):
          self.new_emb_group_deref[i] = self.all_embs[new_emb_group[i]]
    
      if failed_narrowdown:
        continue

      best_score = self.best_emb_groups[-1][0][0]
      score_improvement = self.last_best_score / best_score

      keylist = list(self.cur_optimization_choices.keys())
      for key_id in range(len(keylist)):
        key = keylist[key_id]
        val = self.cur_optimization_choices[key]

      self.evaluate_optimization_methodology(last_time, score_improvement)
      if best_score < self.last_best_score:
        self.print_best([(self.best_emb_groups, self.iterations)], True)
#        if self.SAVE_ALL == True and len(self.best_emb_groups[self.MAX_SIMILAR_EMBS - 1]) != 0:
#          self.save_best([self.best_emb_groups[self.MAX_SIMILAR_EMBS - 1][0][1]], iterations=self.iterations, score = best_score)

      self.last_best_score = best_score
    
      self.iterations += 1

    self.textbox += self.time_str() + f"<Exiting loop>                             \r"
    self.do_interrupt = False
#    self.save_state()
    self.save_near_emb_cache()
    
    self.textbox += self.time_str() + f"<Run complete>                             \r"
    return self.best_emb_groups, self.iterations



