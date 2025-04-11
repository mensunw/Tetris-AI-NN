package src.pas.tetris.agents;

import java.util.Arrays;
// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.lang.Math;

// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.game.Block;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU; // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;

public class TetrisQAgent
        extends QAgent {

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() {
        return this.random;
    }

    @Override
    public Model initQFunction() {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        final int numPixelsInImage = (Board.NUM_ROWS * Board.NUM_COLS) + 6;
        final int hiddenDim = (2 * numPixelsInImage) + 6;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /**
     * This function is for you to figure out what your features
     * are. This should end up being a single row-vector, and the
     * dimensions should be what your qfunction is expecting.
     * One thing we can do is get the grayscale image
     * where squares in the image are 0.0 if unoccupied, 0.5 if
     * there is a "background" square (i.e. that square is occupied
     * but it is not the current piece being placed), and 1.0 for
     * any squares that the current piece is being considered for.
     * 
     * We can then flatten this image to get a row-vector, but we
     * can do more than this! Try to be creative: how can you measure the
     * "state" of the game without relying on the pixels? If you were given
     * a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
            final Mino potentialAction) {
        Matrix vector = null;
        try {
            Board board = game.getBoard();
            // Grayscale image
            Matrix grayScale = game.getGrayscaleImage(potentialAction);
            // Flattened Image
            // System.out.println(grayScale);
            Matrix flattenedImage = grayScale.flatten();
            // think about way to summarize flattened image, too many features (also account

            // Check if player lost
            double agentLost = 0.0;
            if (game.didAgentLose()) {
                agentLost = 1.0;
            }

            // Check total score for entire game
            double totalScore = game.getTotalScore();

            // Check the score for placing mino
            double rowsFilled = 0.0;
            int rows = 22;
            int cols = 10;
            for (int row = 0; row < rows; row++) {
                // Search for full rows
                boolean filled = true;
                for (int col = 0; col < cols; col++) {
                    if (grayScale.get(row, col) == 0.0) {
                        filled = false;
                        break;
                    }
                }
                if (filled) {
                    rowsFilled += 1.0;
                }
            }

            // Find holes
            double holes = 0.0;
            for (int col = 0; col < cols; col++) {
                boolean blockFound = false;
                for (int row = 0; row < rows; row++) {
                    if (grayScale.get(row, col) != 0.0) {
                        // Found block
                        blockFound = true;
                    } else if (blockFound && grayScale.get(row, col) == 0.0) {
                        // Block has been found but there's an empty space below it
                        holes += 1.0;
                    }
                }
            }

            // Find height
            double maxHeight = 0.0;
            double[] heights = new double[cols];
            for (int col = 0; col < cols; col++) {
                for (int row = 0; row < rows; row++) {
                    if (grayScale.get(row, col) != 0.0) {
                        // Found block, so this is the height
                        int height = (rows - row);
                        heights[col] = height;
                        if (height > maxHeight) {
                            maxHeight = height;
                        }
                        break;
                    }
                }
            }

            // Calculate average height
            double totalHeight = 0.0;
            for (double height : heights) {
                totalHeight += height;
            }
            double avgHeight = totalHeight / cols;

            // Standard deviation
            double stdDev = 0.0;
            for (double height : heights) {
                stdDev += Math.pow(height - avgHeight, 2);
            }
            stdDev = Math.sqrt(stdDev / cols);

            // Thread.sleep(1000);
            // System.out.println("stdDev:" + stdDev);
            // Create matrix for these features
            Matrix features = Matrix.full(1, 6, 0.0);
            features.set(0, 0, agentLost);
            features.set(0, 1, rowsFilled);
            features.set(0, 2, totalScore);
            features.set(0, 3, holes);
            features.set(0, 4, maxHeight);
            features.set(0, 5, stdDev);

            // Combine both matrices for all features
            int totalCols = flattenedImage.numel() + features.numel();
            vector = Matrix.full(1, totalCols, 0.0);

            // Copy grayscale image into new vector
            for (int index = 0; index < flattenedImage.numel(); index++) {
                vector.set(0, index, flattenedImage.get(0, index));
            }

            // Copy numerical features into new vector
            for (int index = 0; index < features.numel(); index++) {
                vector.set(0, flattenedImage.numel() + index, features.get(0, index));
            }
            return vector;
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return vector;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good"
     * actions
     * over and over again. This can prevent us from discovering new, potentially
     * even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of
     * the time.
     * While this strategy is easy to implement, it often doesn't perform well and
     * is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
            final GameCounter gameCounter) {
        // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() +
        // "\tgameIdx=" + gameCounter.getCurrentGameIdx());
        double phaseIdx = gameCounter.getCurrentPhaseIdx();
        // Start with high exploration
        double INITIAL_EXPLORATION = 0.7;

        /*
         * if (phaseIdx >= 300.0){
         * INITIAL_EXPLORATION = 0.0;
         * }
         */
        // Minimum exploration probability
        double MIN_EXPLORATION = 0.05;
        // Rate at which exploration decays
        double DECAY_RATE = 0.005;

        // Calculate decayed exploration probability
        long gameIdx = gameCounter.getCurrentGameIdx();
        double explorationProb = Math.max(MIN_EXPLORATION,
                INITIAL_EXPLORATION * Math.exp(-DECAY_RATE * gameIdx));

        // System.out.println("exploreprob: " + explorationProb);
        // Decide whether to explore or exploit
        // System.out.println("shouldExplore: " + (this.getRandom().nextDouble() <=
        // explorationProb));

        return this.getRandom().nextDouble() <= explorationProb;
        // return this.getRandom().nextDouble() <= INITIAL_EXPLORATION;
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we
     * decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game) {
        int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());

        // Get highest y coordinate for each block for each mino
        int lowestMino = -1;
        int lowestMinoCount = 0;
        Mino ret = game.getFinalMinoPositions().get(randIdx);
        for (Mino mino : game.getFinalMinoPositions()) {
            Block[] blocks = mino.getBlocks();

            // get lowest coordinate for each mino, and count it
            int lowestBlock = -1;
            int lowestBlockCount = 0;
            for (Block block : blocks) {
                int y_cord = block.getCoordinate().getYCoordinate();
                // If found lower y coordinate found, re-set lowest and re-set count
                if (y_cord > lowestBlock) {
                    lowestBlock = y_cord;
                    lowestBlockCount = 1;
                    // If found same height as lowest, inc count
                } else if (y_cord == lowestBlock) {
                    lowestBlockCount += 1;
                }
            }
            if (lowestBlock > lowestMino) {
                lowestMino = lowestBlock;
                lowestMinoCount = lowestBlockCount;
                ret = mino;
            } else if ((lowestBlock == lowestMino) && (lowestBlockCount > lowestMinoCount)) {
                lowestMino = lowestBlock;
                lowestMinoCount = lowestBlockCount;
                ret = mino;
            }
        }
        // System.out.println("lowestMino: " + lowestMino);

        // Low chancee to ignore
        if (Math.random() < 0.05) {
            return game.getFinalMinoPositions().get(randIdx);
        }
        return ret;
    }

    /**
     * This method is called by the TrainerAgent after we have played enough
     * training games.
     * In between the training section and the evaluation section of a phase, we
     * need to use
     * the exprience we've collected (from the training games) to improve the
     * q-function.
     *
     * You don't really need to change this method unless you want to. All that
     * happens
     * is that we will use the experiences currently stored in the replay buffer to
     * update
     * our model. Updates (i.e. gradient descent updates) will be applied per
     * minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient
     * descent manner
     * (i.e. all at once)...this often works better and is an active area of
     * research.
     *
     * Each pass through the data is called an epoch, and we will perform
     * "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
            LossFunction lossFunction,
            Optimizer optimizer,
            long numUpdates) {
        for (int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix>> batchIterator = dataset.iterator();

            while (batchIterator.hasNext()) {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                            lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the
     * larger
     * the number, the more "pleasurable" it is to the model, and the smaller the
     * number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced
     * by the
     * points, however this is not all. In fact, just using the points earned this
     * turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and
     * "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally
     * the worse
     * (unless you have a long hole waiting for an I-block). When you design a
     * reward
     * signal that is less sparse, you should see your model optimize this reward
     * over time.
     */
    @Override
    public double getReward(final GameView game) {
        try {
            // Terminal state is no good.
            if (game.didAgentLose()) {
                return -1000.0;
            }

            // Give some (and by some i mean alot) reward for clearing lines
            double linesR = game.getScoreThisTurn() * 300;

            // These are the dimensions I found when printing, also can be found by
            // board.NUM_ROWS & board.NUM_COLS
            Board board = game.getBoard();
            int rows = 22;
            int cols = 10;

            // Find holes and penalize
            double holesP = 0.0;
            for (int col = 0; col < cols; col++) {
                boolean blockFound = false;
                for (int row = 0; row < rows; row++) {
                    if (board.getBlockAt(col, row) != null) {
                        // Found block
                        blockFound = true;
                    } else if (blockFound && board.getBlockAt(col, row) == null) {
                        // Block has been found but there's an empty space below it
                        holesP -= 0.25;
                    }
                }
            }

            // Find height
            double maxHeight = 0.0;
            double[] heights = new double[cols];
            for (int col = 0; col < cols; col++) {
                for (int row = 0; row < rows; row++) {
                    if (board.getBlockAt(col, row) != null) {
                        // Found block, so this is the height
                        int height = (rows - row);
                        heights[col] = height;
                        if (height > maxHeight) {
                            maxHeight = height;
                        }
                        break;
                    }
                }
            }

            // Penalty for total height
            double heightP = -0.1 * maxHeight;

            // Calculate average height
            double totalHeight = 0.0;
            for (double height : heights) {
                totalHeight += height;
            }
            double avgHeight = totalHeight / cols;

            // Reward balanced boards by penalizing height standard deviation
            double stdDev = 0.0;
            for (double height : heights) {
                stdDev += Math.pow(height - avgHeight, 2);
            }
            stdDev = Math.sqrt(stdDev / cols);
            double balanceP = -1.5 * stdDev;

            // System.out.println("downR: " + downR);
            // System.out.println("closeR: " + closeR);

            // System.out.println("max height: " + heightP);
            // System.out.println("holes: " + holesP);
            // System.out.println("balance: " + balanceP);
            // Combine points
            double reward = linesR + holesP + heightP + balanceP;
            // System.out.println("reward: " + reward);
            // Thread.sleep(5000);
            return reward;
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return 0.0;
    }

}
