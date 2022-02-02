from game import Game
from agent import Agent


if __name__ == '__main__':
    agent = Agent(0.001, 1000, 64, 0.9, 0.75)
    game = Game(agent)
    game.play(1000, 1000, 10, 10)
    game.save()
    game.plot()
