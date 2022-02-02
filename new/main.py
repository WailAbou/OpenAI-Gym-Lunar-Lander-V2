from game import Game
from agent import Agent


if __name__ == '__main__':
    agent = Agent(0.001, 1000, 64, 0.9, 0.7)
    game = Game(agent, False)
    game.play(1000, 1000, 10, 20)
    game.save()
    game.plot()
