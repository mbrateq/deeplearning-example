package pl.sggw.deeplearningexample.deeplearning4j;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class IrisClassifier {

    private static final int CLASSES_COUNT = 3;
    private static final int FEATURES_COUNT = 4;

    private static final String FILE_NAME = "iris.txt";
    private static final int SKIP_LINES_NUM = 1;
    private static final char DELIMITER = ',';

    public void execute() throws IOException, InterruptedException {

        DataSet allData;
        try (RecordReader recordReader = new CSVRecordReader(SKIP_LINES_NUM, DELIMITER)) {
            recordReader.initialize(new FileSplit(new ClassPathResource(FILE_NAME).getFile()));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            allData = iterator.next();
        }

//      for test purposes
//      allData.shuffle(System.currentTimeMillis());
        allData.shuffle(42);

//      normalize data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

//      split data
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

//        configure neural network
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .iterations(1000)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .learningRate(0.1).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build())
                .backpropType(BackpropType.Standard).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();


//        UI config
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

//      start training
        model.fit(trainingData);

//      evaluate data
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);

        System.out.println(eval.stats());

    }

}
