from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

if __name__ == "__main__":

    # checker_board = Checker(1000, 50)
    # checker_board.draw()
    # checker_board.show()
    
    generator = ImageGenerator("./exercise_data/", "Labels.json", 
                               5, [32,32, 3], True, True)
    generator.next()
    generator.show()
    # circle = Circle(6250, 250, (3125, 3125))
    # circle.draw()
    # circle.show()

    # spectrum = Spectrum(5)
    # spectrum.draw()
    # spectrum.show()

    # #test checkers
    # test_checkers = TestCheckers()
    # test_checkers.setUp()
    # test_checkers.testPattern()
    # test_checkers.testPatternDifferentSize()
    # test_checkers.testReturnCopy

    # #test circle
    # test_circle = TestCircle()
    # test_circle.setUp()
    # test_circle.testPattern()
    # test_circle.testPatternDifferentSize()
    # test_circle.testReturnCopy

    # test spectrum
    # test_spectrum = TestSpectrum()
    # test_spectrum.setUp()
    # test_spectrum.testPattern()
    # test_spectrum.testPatternDifferentSize()
    # test_spectrum.testReturnCopy
