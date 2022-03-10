To reproduce the result in Figure 1, run 
python alphazero.py --game=CartPole-v1 --algorithm=uct --epsilon=0.1
python alphazero.py --game=CartPole-v1 --algorithm=ments --epsilon=0.1
python alphazero.py --game=CartPole-v1 --algorithm=rents --epsilon=0.1
python alphazero.py --game=CartPole-v1 --algorithm=tsallis --epsilon=0.1

python alphazero.py --game=Acrobot-v1 --algorithm=uct --epsilon=0.1
python alphazero.py --game=Acrobot-v1 --algorithm=ments --epsilon=0.1
python alphazero.py --game=Acrobot-v1 --algorithm=rents --epsilon=0.1
python alphazero.py --game=Acrobot-v1 --algorithm=tsallis --epsilon=0.1

To reproduce results in Table 2:
to run MENTS:
python alphago.py --game=AsterixNoFrameskip-v4 --tau=0.02 --algorithm=ments --epsilon=0.

Replace algorithm and game parameters with the following:
Please note that due to the maximum size allowed to submit, we have to remove 7 models from the 
model folder.
List of games:
    AlienNoFrameskip-v4,
    AmidarNoFrameskip-v4,
    AsteroidsNoFrameskip-v4,
    BankHeistNoFrameskip-v4,
    BowlingNoFrameskip-v4,
    CentipedeNoFrameskip-v4,
    DemonAttackNoFrameskip-v4,
    GopherNoFrameskip-v4,
    PhoenixNoFrameskip-v4,
    RobotankNoFrameskip-v4,
    WizardOfWorNoFrameskip-v4,
    AtlantisNoFrameskip-v4,
    EnduroNoFrameskip-v4,
    FrostbiteNoFrameskip-v4,
    HeroNoFrameskip-v4,
    MsPacmanNoFrameskip-v4,
    SolarisNoFrameskip-v4,
    BreakoutNoFrameskip-v4,
    AsterixNoFrameskip-v4,
    BeamRiderNoFrameskip-v4,
    QbertNoFrameskip-v4,
    SeaquestNoFrameskip-v4,
    SpaceInvadersNoFrameskip-v4
List of algorithm:
    uct (parameter: c)
    power-uct (parameter: c and p)
    maxmcts(parameter: lambda_const)
    rents (parameter: tau and epsilon)
    ments (parameter: tau and epsilon)
# w_mcts_atari
