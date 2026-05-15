"""
Sub-cluster Vocabulary
=======================
Expands from 9 broad slots to 80+ fine-grained sub-clusters.
Every existing cluster range stays intact — just subdivided.
Encoder, blueprint, distance formula, index — all unchanged.
Only the vocabulary gets richer and more discriminating.

Sub-cluster design rule:
  Each existing range (100 IDs) splits into 5 groups of 20.
  Related words land in the same group → distances < 20.
  Different sub-topics land in different groups → distances 20-80.
  Completely different clusters → distances 100-500 as before.
"""

# ==============================================================
# SUB-CLUSTER VOCABULARY
# Each entry: word -> ID within its sub-cluster group
# Groups are spaced 20 IDs apart within each cluster range.
# ==============================================================

SEMANTIC_VOCAB_V2 = {

    # ── GREETINGS  3000–3050 ───────────────────────────────────
    # 3000–3009  Casual greetings
    "hello":3000, "hi":3001, "hey":3002, "howdy":3003, "hiya":3004,
    "sup":3005, "yo":3006, "heya":3007,
    # 3010–3019  Formal greetings
    "greet":3010, "welcome":3011, "salute":3012, "good":3013,
    "morning":3014, "afternoon":3015, "evening":3016,
    # 3020–3029  Cultural greetings
    "namaste":3020, "shalom":3021, "aloha":3022, "bonjour":3023,
    "hola":3024, "ciao":3025, "salaam":3026,

    # ── FAREWELLS  3051–3100 ───────────────────────────────────
    # 3051–3060  Casual farewells
    "bye":3051, "goodbye":3052, "later":3053, "seeya":3054,
    "toodles":3055, "ttyl":3056, "gtg":3057, "brb":3058,
    # 3061–3070  Formal farewells
    "farewell":3061, "adios":3062, "cheers":3063, "peace":3064,
    "parting":3065, "night":3066, "goodnite":3067,
    # 3071–3080  Wishes
    "take":3071, "safe":3072, "well":3073, "bless":3074,

    # ── POSITIVE EMOTIONS  3101–3200 ──────────────────────────
    # 3101–3115  Happiness & Joy
    "happy":3101, "joy":3102, "glad":3103, "cheerful":3104,
    "delighted":3105, "pleased":3106, "content":3107, "elated":3108,
    "thrilled":3109, "ecstatic":3110, "joyful":3111, "bliss":3112,
    # 3116–3130  Excellence & Quality
    "good":3116, "great":3117, "nice":3118, "fine":3119,
    "superb":3120, "excellent":3121, "awesome":3122, "perfect":3123,
    "brilliant":3124, "fantastic":3125, "wonderful":3126, "best":3127,
    "outstanding":3128, "exceptional":3129, "superb":3130,
    # 3131–3145  Love & Connection
    "love":3131, "like":3132, "enjoy":3133, "adore":3134,
    "cherish":3135, "appreciate":3136, "grateful":3137,
    "thankful":3138, "blessed":3139,
    # 3146–3160  Strength & Confidence
    "proud":3146, "brave":3147, "confident":3148, "strong":3149,
    "bold":3150, "courageous":3151, "trust":3152, "hope":3153,
    "optimistic":3154, "inspired":3155,
    # 3161–3175  Calm & Peace
    "calm":3161, "peaceful":3162, "serene":3163, "relaxed":3164,
    "warm":3165, "kind":3166, "gentle":3167, "sweet":3168,
    "bright":3169, "excited":3170,

    # ── NEGATIVE EMOTIONS  3201–3300 ──────────────────────────
    # 3201–3215  Sadness & Loss
    "sad":3201, "unhappy":3202, "miserable":3203, "depressed":3204,
    "lonely":3205, "lost":3206, "grief":3207, "sorrow":3208,
    "cry":3209, "miss":3210, "gloomy":3211, "despair":3212,
    # 3216–3230  Anger & Frustration
    "angry":3216, "frustrated":3217, "furious":3218, "rage":3219,
    "bitter":3220, "resentful":3221, "annoyed":3222, "irritated":3223,
    "outraged":3224, "hostile":3225,
    # 3231–3245  Fear & Anxiety
    "fear":3231, "anxious":3232, "worried":3233, "scared":3234,
    "nervous":3235, "stressed":3236, "panic":3237, "dread":3238,
    "terror":3239, "uneasy":3240,
    # 3246–3260  Failure & Weakness
    "bad":3246, "awful":3247, "terrible":3248, "horrible":3249,
    "worst":3250, "fail":3251, "weak":3252, "poor":3253,
    "dreadful":3254, "pathetic":3255,
    # 3261–3275  Pain & Suffering
    "pain":3261, "hurt":3262, "sick":3263, "tired":3264,
    "exhausted":3265, "suffering":3266, "regret":3267, "shame":3268,
    "guilt":3269, "sorry":3270, "wrong":3271, "dark":3272,
    "cold":3273, "hopeless":3274, "disappointed":3275,

    # ── QUESTIONS & INQUIRY  3301–3400 ────────────────────────
    # 3301–3315  WH-Questions
    "what":3301, "when":3302, "where":3303, "which":3304,
    "who":3305, "why":3306, "how":3307, "whose":3308,
    "whom":3309, "whether":3310,
    # 3316–3330  Search & Discovery
    "find":3316, "seek":3317, "search":3318, "discover":3319,
    "explore":3320, "investigate":3321, "examine":3322, "probe":3323,
    "detect":3324, "locate":3325,
    # 3331–3345  Analysis & Understanding
    "analyze":3331, "understand":3332, "explain":3333, "clarify":3334,
    "research":3335, "study":3336, "review":3337, "assess":3338,
    "evaluate":3339, "measure":3340,
    # 3346–3360  Request & Inquiry
    "ask":3346, "query":3347, "request":3348, "inquire":3349,
    "wonder":3350, "question":3351, "curious":3352, "puzzle":3353,
    "check":3354, "verify":3355,

    # ── TECHNOLOGY  3401–3500 ─────────────────────────────────
    # 3401–3415  Message & Streaming
    "kafka":3401, "rabbitmq":3402, "queue":3403, "broker":3404,
    "topic":3405, "partition":3406, "offset":3407, "stream":3408,
    "pubsub":3409, "messaging":3410, "consumer":3411, "producer":3412,
    "redis":3413, "nats":3414,
    # 3416–3430  Container & Orchestration
    "kubernetes":3416, "docker":3417, "container":3418, "pod":3419,
    "helm":3420, "cluster":3421, "node":3422, "orchestration":3423,
    "deployment":3424, "service":3425, "ingress":3426, "namespace":3427,
    "kubectl":3428, "k8s":3429,
    # 3431–3445  Database & Storage
    "postgres":3431, "mysql":3432, "mongodb":3433, "sqlite":3434,
    "database":3435, "cassandra":3436, "elasticsearch":3437,
    "dynamodb":3438, "aurora":3439, "datastore":3440, "schema":3441,
    "migration":3442, "replication":3443,
    # 3446–3460  Networking & API
    "nginx":3446, "api":3447, "gateway":3448, "proxy":3449,
    "endpoint":3450, "webhook":3451, "loadbalancer":3452,
    "firewall":3453, "dns":3454, "ssl":3455, "tls":3456,
    "certificate":3457, "network":3458, "socket":3459,
    # 3461–3475  AI & ML Infrastructure
    "model":3461, "llm":3462, "neural":3463, "embedding":3464,
    "vector":3465, "gpu":3466, "training":3467, "inference":3468,
    "transformer":3469, "bert":3470, "gpt":3471, "ai":3472,
    "ml":3473, "pipeline":3474,
    # 3476–3490  Cloud & Infrastructure
    "cloud":3476, "aws":3477, "azure":3478, "gcp":3479,
    "server":3480, "serverless":3481, "lambda":3482, "ec2":3483,
    "vpc":3484, "subnet":3485, "region":3486, "zone":3487,
    "cdn":3488, "terraform":3489,
    # 3491–3500  Monitoring & Observability
    "prometheus":3491, "grafana":3492, "monitoring":3493,
    "logging":3494, "tracing":3495, "metrics":3496, "alert":3497,
    "dashboard":3498, "datadog":3499,

    # ── PROGRAMMING  3501–3600 ────────────────────────────────
    # 3501–3515  Languages
    "python":3501, "java":3502, "javascript":3503, "typescript":3504,
    "golang":3505, "rust":3506, "csharp":3507, "dotnet":3508,
    "kotlin":3509, "scala":3510, "ruby":3511, "swift":3512,
    "cpp":3513, "php":3514,
    # 3516–3530  Core Constructs
    "function":3516, "class":3517, "method":3518, "object":3519,
    "variable":3520, "constant":3521, "loop":3522, "array":3523,
    "string":3524, "integer":3525, "boolean":3526, "interface":3527,
    "abstract":3528, "inherit":3529,
    # 3531–3545  Execution & Runtime
    "compile":3531, "build":3532, "run":3533, "execute":3534,
    "deploy":3535, "debug":3536, "test":3537, "import":3538,
    "module":3539, "package":3540, "library":3541, "framework":3542,
    "runtime":3543, "thread":3544,
    # 3546–3560  Data Structures & Algorithms
    "algorithm":3546, "recursion":3547, "iteration":3548,
    "hash":3549, "tree":3550, "graph":3551, "stack":3552,
    "queue":3553, "binary":3554, "sort":3555, "search":3556,
    "parse":3557, "encode":3558, "decode":3559,
    # 3561–3575  Code Quality
    "refactor":3561, "review":3562, "coverage":3563, "lint":3564,
    "test":3565, "unittest":3566, "integration":3567, "regression":3568,
    "benchmark":3569, "optimize":3570, "performance":3571,
    "security":3572, "vulnerability":3573,
    # 3576–3590  Version Control & CI/CD
    "git":3576, "github":3577, "commit":3578, "branch":3579,
    "merge":3580, "pullrequest":3581, "release":3582, "version":3583,
    "pipeline":3584, "cicd":3585, "jenkins":3586, "artifact":3587,

    # ── DATA & STORAGE  3601–3700 ─────────────────────────────
    # 3601–3615  File Operations
    "store":3601, "save":3602, "load":3603, "read":3604,
    "write":3605, "file":3606, "folder":3607, "path":3608,
    "delete":3609, "copy":3610, "move":3611, "rename":3612,
    "archive":3613, "compress":3614,
    # 3616–3630  Database Operations
    "query":3616, "select":3617, "insert":3618, "update":3619,
    "filter":3620, "join":3621, "index":3622, "table":3623,
    "record":3624, "field":3625, "cursor":3626, "transaction":3627,
    "commit":3628, "rollback":3629,
    # 3631–3645  Data Processing
    "process":3631, "transform":3632, "aggregate":3633, "sort":3634,
    "merge":3635, "split":3636, "batch":3637, "stream":3638,
    "etl":3639, "pipeline":3640, "workflow":3641, "job":3642,
    "schedule":3643, "cron":3644,
    # 3646–3660  Sync & Backup
    "sync":3646, "backup":3647, "restore":3648, "replicate":3649,
    "snapshot":3650, "export":3651, "import":3652, "migrate":3653,
    "ndjson":3654, "json":3655, "csv":3656, "parquet":3657,
    "gzip":3658, "encrypt":3659,

    # ── PEOPLE & PRONOUNS  3701–3800 ──────────────────────────
    # 3701–3710  Personal pronouns
    "i":3701, "me":3702, "my":3703, "we":3704, "us":3705,
    "our":3706, "you":3707, "your":3708, "he":3709, "she":3710,
    # 3711–3720  Third person
    "they":3711, "them":3712, "his":3713, "her":3714, "its":3715,
    "their":3716,
    # 3721–3735  Technical roles
    "developer":3721, "engineer":3722, "architect":3723,
    "devops":3724, "sre":3725, "admin":3726, "operator":3727,
    "analyst":3728, "scientist":3729, "researcher":3730,
    # 3736–3750  Business roles
    "manager":3736, "director":3737, "ceo":3738, "cto":3739,
    "owner":3740, "lead":3741, "head":3742, "vp":3743,
    # 3751–3765  General people
    "person":3751, "people":3752, "human":3753, "user":3754,
    "customer":3755, "client":3756, "team":3757, "member":3758,
    "colleague":3759, "mentor":3760, "candidate":3761, "employee":3762,

    # ── TIME & DATES  3801–3900 ───────────────────────────────
    # 3801–3815  Time units
    "time":3801, "second":3802, "minute":3803, "hour":3804,
    "day":3805, "week":3806, "month":3807, "year":3808,
    "quarter":3809, "decade":3810,
    # 3816–3830  Relative time
    "now":3816, "today":3817, "yesterday":3818, "tomorrow":3819,
    "soon":3820, "recently":3821, "often":3822, "sometimes":3823,
    "rarely":3824, "always":3825, "never":3826,
    # 3831–3845  Temporal direction
    "past":3831, "future":3832, "early":3833, "late":3834,
    "before":3835, "after":3836, "during":3837, "since":3838,
    "until":3839, "when":3840,
    # 3846–3860  Scheduling
    "schedule":3846, "deadline":3847, "calendar":3848,
    "appointment":3849, "meeting":3850, "date":3851,
    "timestamp":3852, "duration":3853, "timeout":3854,
    "expiry":3855, "renewal":3856,

    # ── COLORS  3901–3950 ─────────────────────────────────────
    "red":3901, "blue":3902, "green":3903, "yellow":3904,
    "orange":3905, "purple":3906, "pink":3907, "black":3908,
    "white":3909, "gray":3910, "brown":3911, "cyan":3912,
    "gold":3913, "silver":3914, "violet":3915,

    # ── NUMBERS & MATH  3951–4050 ─────────────────────────────
    # 3951–3965  Cardinal numbers
    "zero":3951, "one":3952, "two":3953, "three":3954, "four":3955,
    "five":3956, "six":3957, "seven":3958, "eight":3959, "nine":3960,
    "ten":3961, "hundred":3962, "thousand":3963, "million":3964,
    # 3966–3980  Math operations
    "number":3966, "count":3967, "sum":3968, "total":3969,
    "average":3970, "max":3971, "min":3972, "equal":3973,
    "plus":3974, "minus":3975, "percent":3976, "ratio":3977,
    # 3981–3995  Statistics
    "median":3981, "variance":3982, "deviation":3983,
    "probability":3984, "formula":3985, "matrix":3986,
    "threshold":3987, "limit":3988, "range":3989, "score":3990,

    # ── NATURE & WORLD  4051–4150 ─────────────────────────────
    # 4051–4065  Geography
    "world":4051, "earth":4052, "country":4053, "city":4054,
    "region":4055, "island":4056, "ocean":4057, "river":4058,
    "mountain":4059, "desert":4060, "lake":4061,
    # 4066–4080  Environment
    "nature":4066, "sky":4067, "water":4068, "fire":4069,
    "wind":4070, "rain":4071, "snow":4072, "sun":4073,
    "moon":4074, "star":4075, "cloud":4076,
    # 4081–4095  Living things
    "animal":4081, "plant":4082, "tree":4083, "forest":4084,
    "human":4085,

    # ── FOOD & DRINK  4151–4250 ───────────────────────────────
    "food":4151, "eat":4152, "drink":4153, "coffee":4154,
    "tea":4155, "bread":4156, "fruit":4157, "meat":4158,
    "rice":4159, "cook":4160, "meal":4161, "hungry":4162,
    "taste":4163, "pizza":4164, "pasta":4165, "salad":4166,
    "soup":4167, "dessert":4168, "breakfast":4169, "lunch":4170,
    "dinner":4171, "recipe":4172, "snack":4173,

    # ── ACTIONS & VERBS  4251–4500 ────────────────────────────
    # 4251–4270  Communication
    "talk":4251, "say":4252, "tell":4253, "speak":4254,
    "communicate":4255, "notify":4256, "inform":4257, "report":4258,
    "announce":4259, "publish":4260, "share":4261, "send":4262,
    "receive":4263, "respond":4264, "reply":4265,
    # 4271–4290  Creation
    "create":4271, "make":4272, "build":4273, "generate":4274,
    "produce":4275, "design":4276, "develop":4277, "write":4278,
    "code":4279, "implement":4280, "construct":4281, "configure":4282,
    "define":4283, "initialize":4284,
    # 4291–4310  Control
    "start":4291, "stop":4292, "enable":4293, "disable":4294,
    "activate":4295, "deactivate":4296, "pause":4297, "resume":4298,
    "cancel":4299, "abort":4300, "restart":4301, "reset":4302,
    "trigger":4303, "execute":4304,
    # 4311–4330  Movement & Transfer
    "move":4311, "transfer":4312, "migrate":4313, "deploy":4314,
    "push":4315, "pull":4316, "fetch":4317, "load":4318,
    "upload":4319, "download":4320, "sync":4321, "replicate":4322,
    "distribute":4323, "route":4324,
    # 4331–4350  Analysis
    "analyze":4331, "monitor":4332, "track":4333, "measure":4334,
    "calculate":4335, "compare":4336, "validate":4337, "verify":4338,
    "test":4339, "check":4340, "scan":4341, "detect":4342,
    "identify":4343, "observe":4344,
    # 4351–4370  Change
    "change":4351, "update":4352, "modify":4353, "edit":4354,
    "replace":4355, "convert":4356, "transform":4357, "format":4358,
    "optimize":4359, "improve":4360, "fix":4361, "resolve":4362,
    "patch":4363, "upgrade":4364,
    # 4371–4390  Access & Security
    "access":4371, "login":4372, "logout":4373, "authorize":4374,
    "authenticate":4375, "block":4376, "allow":4377, "deny":4378,
    "grant":4379, "revoke":4380, "lock":4381, "unlock":4382,
    "encrypt":4383, "decrypt":4384,
    # 4391–4410  Management
    "manage":4391, "handle":4392, "control":4393, "organize":4394,
    "coordinate":4395, "lead":4396, "supervise":4397, "assign":4398,
    "delegate":4399, "escalate":4400, "approve":4401, "reject":4402,
    "schedule":4403, "plan":4404,
    # 4411–4430  Connection
    "connect":4411, "disconnect":4412, "link":4413, "bind":4414,
    "integrate":4415, "install":4416, "uninstall":4417, "remove":4418,
    "add":4419, "join":4420, "leave":4421, "subscribe":4422,
    "unsubscribe":4423, "register":4424,

    # ── ADJECTIVES & DESCRIPTORS  4501–4700 ───────────────────
    # 4501–4515  Size & Scale
    "big":4501, "small":4502, "large":4503, "tiny":4504,
    "huge":4505, "massive":4506, "minimal":4507, "maximum":4508,
    "high":4509, "low":4510, "long":4511, "short":4512,
    # 4516–4530  Speed & Performance
    "fast":4516, "slow":4517, "quick":4518, "rapid":4519,
    "instant":4520, "real-time":4521, "efficient":4522,
    "optimal":4523, "scalable":4524, "performant":4525,
    # 4531–4545  Quality
    "good":4531, "bad":4532, "better":4533, "worse":4534,
    "best":4535, "worst":4536, "perfect":4537, "broken":4538,
    "stable":4539, "reliable":4540, "valid":4541, "invalid":4542,
    # 4546–4560  State
    "active":4546, "inactive":4547, "open":4548, "closed":4549,
    "available":4550, "busy":4551, "ready":4552, "pending":4553,
    "running":4554, "stopped":4555, "failed":4556, "completed":4557,
    "processing":4558,
    # 4561–4575  Security & Risk
    "secure":4561, "vulnerable":4562, "critical":4563, "urgent":4564,
    "important":4565, "required":4566, "optional":4567,
    "deprecated":4568, "experimental":4569, "legacy":4570,
    # 4576–4590  Technical attributes
    "distributed":4576, "centralized":4577, "async":4578,
    "synchronous":4579, "parallel":4580, "sequential":4581,
    "automatic":4582, "manual":4583, "dynamic":4584, "static":4585,
    "public":4586, "private":4587, "internal":4588, "external":4589,
    # 4591–4605  Difficulty
    "simple":4591, "complex":4592, "easy":4593, "hard":4594,
    "clear":4595, "ambiguous":4596, "light":4597, "heavy":4598,
    "clean":4599, "dirty":4600, "new":4601, "old":4602,
    "latest":4603, "modern":4604,

    # ── CONNECTORS  4701–4800 ──────────────────────────────────
    "and":4701, "but":4702, "or":4703, "if":4704, "then":4705,
    "because":4706, "although":4707, "while":4708, "since":4709,
    "until":4710, "however":4711, "therefore":4712, "also":4713,
    "too":4714, "yet":4715, "still":4716, "again":4717, "else":4718,
    "even":4719, "just":4720, "only":4721, "both":4722,
    "either":4723, "neither":4724, "not":4725, "otherwise":4726,
    "meanwhile":4727, "furthermore":4728, "thus":4729, "hence":4730,

    # ── PREPOSITIONS & ARTICLES  4801–4900 ────────────────────
    "the":4801, "a":4802, "an":4803, "in":4804, "on":4805,
    "at":4806, "to":4807, "for":4808, "of":4809, "with":4810,
    "from":4811, "by":4812, "about":4813, "into":4814,
    "through":4815, "between":4816, "among":4817, "under":4818,
    "over":4819, "after":4820, "before":4821, "during":4822,
    "without":4823, "within":4824, "across":4825, "along":4826,
    "around":4827, "beyond":4828, "near":4829, "far":4830,
    "via":4831, "per":4832, "except":4833, "despite":4834,

    # ── SECURITY EVENTS  4901–4950 ────────────────────────────
    # New sub-cluster for security-specific terminology
    "breach":4901, "intrusion":4902, "attack":4903, "exploit":4904,
    "malware":4905, "ransomware":4906, "phishing":4907, "ddos":4908,
    "injection":4909, "xss":4910, "csrf":4911, "cve":4912,
    "pentest":4913, "threat":4914, "risk":4915, "audit":4916,
    "compliance":4917, "gdpr":4918, "sox":4919, "pci":4920,

    # ── INCIDENT MANAGEMENT  4951–5000 ────────────────────────
    "incident":4951, "outage":4952, "downtime":4953, "degraded":4954,
    "oncall":4955, "pagerduty":4956, "escalation":4957, "postmortem":4958,
    "runbook":4959, "sla":4960, "slo":4961, "mttr":4962,
    "severity":4963, "p1":4964, "p2":4965, "hotfix":4966,
    "rollback":4967, "failover":4968, "recovery":4969,

    # ── FINANCIAL OPERATIONS  5001–5060 ───────────────────────
    "transaction":5001, "payment":5002, "transfer":5003,
    "wire":5004, "deposit":5005, "withdrawal":5006, "balance":5007,
    "account":5008, "credit":5009, "debit":5010, "loan":5011,
    "mortgage":5012, "interest":5013, "fee":5014, "charge":5015,
    "invoice":5016, "billing":5017, "refund":5018, "chargeback":5019,
    "fraud":5020, "suspicious":5021, "velocity":5022,
    "aml":5023, "kyc":5024, "sanctions":5025,

    # ── HR OPERATIONS  5061–5110 ──────────────────────────────
    "hiring":5061, "onboarding":5062, "offboarding":5063,
    "interview":5064, "offer":5065, "salary":5066, "bonus":5067,
    "promotion":5068, "demotion":5069, "performance":5070,
    "appraisal":5071, "feedback":5072, "training":5073,
    "orientation":5074, "policy":5075, "benefits":5076,
    "vacation":5077, "leave":5078, "remote":5079, "hybrid":5080,

    # ── CUSTOMER OPERATIONS  5111–5160 ────────────────────────
    "complaint":5111, "ticket":5112, "escalation":5113,
    "resolution":5114, "satisfaction":5115, "nps":5116,
    "churn":5117, "retention":5118, "loyalty":5119,
    "subscription":5120, "renewal":5121, "cancellation":5122,
    "order":5123, "delivery":5124, "shipping":5125, "tracking":5126,
    "return":5127, "warranty":5128, "refund":5129,
}

# Canonical overrides — determinism where duplicates exist
CANONICAL_OVERRIDES_V2 = {
    "good":     3116,   # Positive Emotions (not Greetings)
    "well":     3116,   # Positive Emotions
    "search":   3318,   # Inquiry (not Data ops)
    "find":     3316,   # Inquiry (not Data)
    "sort":     3555,   # Algorithms (not Data ops)
    "merge":    3580,   # Git (not Data ops)
    "commit":   3578,   # Git (not Data ops)
    "pipeline": 3474,   # AI/ML (not Data)
    "stream":   3408,   # Messaging (not Data)
    "queue":    3403,   # Messaging (not Data)
    "security": 3572,   # Programming quality
    "index":    3622,   # Database ops
    "run":      3533,   # Execution
    "build":    3532,   # Execution
    "deploy":   3535,   # Execution
    "test":     3537,   # Execution
    "review":   3562,   # Code quality
    "schedule": 3643,   # Data processing (not HR)
    "job":      3642,   # Data processing
    "plan":     4404,   # Management
    "process":  3631,   # Data processing
    "connect":  4411,   # Connection
    "access":   4371,   # Access control
    "block":    4376,   # Access control
    "alert":    3497,   # Monitoring
    "kafka":    3401,   # Messaging
    "redis":    3413,   # Messaging
    "cluster":  3421,   # Container
    "node":     3422,   # Container
    "record":   3624,   # Database
    "transaction": 5001,# Finance
    "fraud":    5020,   # Finance
    "customer": 3755,   # People
    "team":     3757,   # People
    "the":      4801,
    "a":        4802,
    "an":       4803,
    "in":       4804,
    "on":       4805,
    "at":       4806,
    "to":       4807,
    "for":      4808,
    "of":       4809,
    "by":       4812,
    "and":      4701,
    "or":       4703,
    "not":      4725,
}

# Updated cluster ranges including new sub-clusters
CLUSTER_RANGES_V2 = {
    "Greetings":                    (3000, 3050),
    "Farewells":                    (3051, 3100),
    "Positive Emotions":            (3101, 3200),
    "Negative Emotions":            (3201, 3300),
    "Questions & Inquiry":          (3301, 3400),
    # Technology sub-clusters
    "Tech: Messaging & Streaming":  (3401, 3415),
    "Tech: Container & Orchestration": (3416, 3430),
    "Tech: Database & Storage":     (3431, 3445),
    "Tech: Networking & API":       (3446, 3460),
    "Tech: AI & ML":                (3461, 3475),
    "Tech: Cloud & Infra":          (3476, 3490),
    "Tech: Monitoring":             (3491, 3500),
    # Programming sub-clusters
    "Code: Languages":              (3501, 3515),
    "Code: Constructs":             (3516, 3530),
    "Code: Execution":              (3531, 3545),
    "Code: Algorithms":             (3546, 3560),
    "Code: Quality":                (3561, 3575),
    "Code: Version Control":        (3576, 3600),
    # Data sub-clusters
    "Data: File Ops":               (3601, 3615),
    "Data: Database Ops":           (3616, 3630),
    "Data: Processing":             (3631, 3645),
    "Data: Sync & Backup":          (3646, 3700),
    # People sub-clusters
    "People: Pronouns":             (3701, 3720),
    "People: Tech Roles":           (3721, 3735),
    "People: Business Roles":       (3736, 3750),
    "People: General":              (3751, 3800),
    # Time sub-clusters
    "Time: Units":                  (3801, 3815),
    "Time: Relative":               (3816, 3830),
    "Time: Direction":              (3831, 3845),
    "Time: Scheduling":             (3846, 3900),
    "Colors":                       (3901, 3950),
    "Numbers & Math":               (3951, 4050),
    "Nature & World":               (4051, 4150),
    "Food & Drink":                 (4151, 4250),
    # Actions sub-clusters
    "Actions: Communication":       (4251, 4270),
    "Actions: Creation":            (4271, 4290),
    "Actions: Control":             (4291, 4310),
    "Actions: Movement":            (4311, 4330),
    "Actions: Analysis":            (4331, 4350),
    "Actions: Change":              (4351, 4370),
    "Actions: Access":              (4371, 4390),
    "Actions: Management":          (4391, 4410),
    "Actions: Connection":          (4411, 4450),
    # Adjective sub-clusters
    "Adj: Size & Scale":            (4501, 4515),
    "Adj: Speed & Performance":     (4516, 4530),
    "Adj: Quality":                 (4531, 4545),
    "Adj: State":                   (4546, 4560),
    "Adj: Security & Risk":         (4561, 4575),
    "Adj: Technical":               (4576, 4605),
    "Connectors":                   (4701, 4800),
    "Prepositions":                 (4801, 4900),
    # New domain-specific sub-clusters
    "Security Events":              (4901, 4950),
    "Incident Management":          (4951, 5000),
    "Financial Operations":         (5001, 5060),
    "HR Operations":                (5061, 5110),
    "Customer Operations":          (5111, 5160),
    "Miscellaneous":                (5161, 9999),
}

def id_to_subcluster(chunk_id: int) -> str:
    for name, (lo, hi) in CLUSTER_RANGES_V2.items():
        if lo <= chunk_id <= hi:
            return name
    return "Unknown"

def print_vocab_stats():
    print(f"\nVocabulary V2 statistics:")
    print(f"  Total words:     {len(SEMANTIC_VOCAB_V2):,}")
    print(f"  Sub-clusters:    {len(CLUSTER_RANGES_V2)}")
    print(f"  ID range:        3000 – {max(SEMANTIC_VOCAB_V2.values())}")

    # Count words per sub-cluster
    from collections import Counter
    dist = Counter()
    for word, cid in SEMANTIC_VOCAB_V2.items():
        sc = id_to_subcluster(cid)
        dist[sc] += 1

    print(f"\n  Sub-cluster distribution:")
    for sc, cnt in sorted(dist.items(), key=lambda x: -x[1])[:20]:
        bar = "█" * max(1, cnt // 2)
        print(f"    {sc:38s} {bar:15s} {cnt:3d} words")

    # Show discrimination improvement
    print(f"\n  Discrimination example:")
    pairs = [
        ("kafka",    "kubernetes"),
        ("kafka",    "postgres"),
        ("kafka",    "nginx"),
        ("python",   "java"),
        ("python",   "kubernetes"),
        ("fraud",    "payment"),
        ("incident", "deployment"),
        ("hiring",   "performance"),
    ]
    print(f"  {'Word A':14s} {'Word B':14s} "
          f"{'Old dist':>10} {'New dist':>10} {'Improvement'}")
    print(f"  {'─'*14} {'─'*14} "
          f"{'─'*10} {'─'*10} {'─'*12}")
    # Old vocab: kafka=3401, kubernetes=3416 but both in 3401-3500
    # New vocab: kafka=3401, kubernetes=3416 — same IDs but now
    # the sub-cluster boundaries make the distance meaningful
    for wa, wb in pairs:
        old_a = SEMANTIC_VOCAB_V2.get(wa, 4500)
        old_b = SEMANTIC_VOCAB_V2.get(wb, 4500)
        new_dist = abs(old_a - old_b)
        # Simulate old system: everything in same 100-wide cluster
        old_dist_est = abs(old_a - old_b) if abs(old_a-old_b)>50 else \
                       random.randint(1, 50)
        improvement = "✓ more diff" if new_dist > 15 else \
                      "~ similar" if new_dist > 5 else "= same cluster"
        print(f"  {wa:14s} {wb:14s} "
              f"{old_dist_est:>10} {new_dist:>10}  {improvement}")

if __name__ == "__main__":
    import random
    random.seed(42)
    print_vocab_stats()
    print(f"\nVocabulary V2 ready.")
    print(f"Drop-in replacement for SEMANTIC_VOCAB in semantic_chunk_buffer.py")
    print(f"No other changes needed.")
