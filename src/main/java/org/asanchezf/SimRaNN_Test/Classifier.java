package org.asanchezf.SimRaNN_Test;

import org.apache.log4j.PropertyConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


public class Classifier {

    private static Logger log = Logger.getLogger(Classifier.class);

    public static void main(String[] args) throws  Exception {

        String configFilename = System.getProperty("user.dir")
                + File.separator + "log4j.properties";
        PropertyConfigurator.configure(configFilename);

        String adaptedRidePath = "/Users/AlbertSanchez/Desktop/Post/WindowedRides/ride.csv";
        String trainedModelPath = "/Users/AlbertSanchez/Desktop/Post/Tests/1/DSNet.zip";
        double filter = 0.95; // Threshold probability to select an incident.
        int maxIncidents = 5; // Number of maximum output incidents that tha Classifier can detect.

        int numClasses = 1;
        int batchSize = rideLength(adaptedRidePath);
        File trainedModelLocation = new File(trainedModelPath);

        // Loading the data to test
        RecordReader recordReader = new CSVRecordReader(0,',');
        recordReader.initialize(new FileSplit(new File(adaptedRidePath)));

        // Build a Input Schema
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsFloat("speed","mean_acc_x","mean_acc_y","mean_acc_z","std_acc_x","std_acc_y","std_acc_z")
                .addColumnDouble("sma")
                .addColumnFloat("mean_svm")
                .addColumnsDouble("entropyX","entropyY","entropyZ")
                .addColumnsInteger("bike_type","phone_location","incident_type")
                .build();

        // Made the necessary transformations
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .integerToOneHot("bike_type",0,8)
                .integerToOneHot("phone_location",0,6)
                .build();

        // Get output schema
        Schema outputSchema = tp.getFinalSchema();

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = outputSchema.getColumnNames().size() - 1; //The label index is in the last column of the schema

        TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,tp);

        // Restore normalizer and the model
        NormalizerStandardize normalizerStandardize = ModelSerializer.restoreNormalizerFromFile(trainedModelLocation);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(trainedModelLocation);

        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(transformProcessRecordReader,batchSize,labelIndex,numClasses);
        normalizerStandardize.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(normalizerStandardize);

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(numClasses);
        Double[] d = new Double[batchSize];

        while(dataSetIterator.hasNext()){
            DataSet next = dataSetIterator.next();
            INDArray output = model.output(next.getFeatures(),false); //get the networks prediction
            System.out.println(output);
            for (int i = 0; i<batchSize; i++){d[i] = output.getDouble(i);}
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        log.info(eval.stats());

        List<Integer> position = new ArrayList(), positives;

        // Save window number if score is > filter
        for (int i=0; i<d.length; i++){ if (d[i]>filter) position.add(i); }

        for(Integer i : position) System.out.println("Position: " + i + " - Score: " + d[i]);
        System.out.println("----");
        System.out.println(position.size());
        System.out.println("----");

        if (position.size()>0)
        {
            positives = obtainPredictedPositiveWindows(position, d, maxIncidents);

            System.out.println("Predicted positives: " + positives.size());
            System.out.println("----");
            System.out.println("Detail:");
            for (Integer i : positives) System.out.println("Position: " + i + " - Score: " + d[i]);
            System.out.println("----");
        }
        else System.out.println("No incidents found in this ride");

    }

    public static int rideLength(String path) throws IOException
    {
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line = br.readLine();
        int lines = 1;

        while (line != null)
        {
            lines++;
            line = br.readLine();
        }
        lines--;
        br.close();

        return lines;

    }

    public static List<Integer> obtainPredictedPositiveWindows(List<Integer> positions, Double[] scores, int maxPositives)
    {
        List<Integer> positives = new ArrayList();
        int prevPos = positions.get(0);
        int consecutivePositiveChoose = prevPos;
        double maxScore = scores[prevPos];
        boolean consecutive = false;

        for (Integer i : positions.subList(1,positions.size()))
        {
            if (prevPos+1 == i) //Consecutive
            {
                consecutive = true;
                if (scores[i]>maxScore)
                {
                    maxScore = scores[i];
                    consecutivePositiveChoose = i;
                }
            }
            else //No Consecutive
            {
                positives.add(consecutivePositiveChoose);
                consecutive = false;
                maxScore = scores[i];
                consecutivePositiveChoose = i;
            }
            prevPos = i;
        }
        positives.add(consecutivePositiveChoose);



        // Delete if more than max number of incidents detected
        if(positives.size()>maxPositives)
        {
            List<Double> tmp_scores = new ArrayList();
            List<Integer> topPositives = new ArrayList();

            System.out.println("Actual positives:");
            System.out.println("Position | Scores");
            for (Integer i : positives) System.out.println(i + " | " + scores[i]);
            System.out.println("------");

            System.out.println("Temporal scores:");
            for(Integer i : positives) tmp_scores.add(scores[i]);
            for(Double d : tmp_scores) System.out.println(d);
            System.out.println("------");

            // Sort descendent score
            Collections.sort(tmp_scores, Collections.reverseOrder());

            // Remove items until size equals to maxPositives
            for (int i=tmp_scores.size()-1; i>=maxPositives; i--) tmp_scores.remove(i);

            System.out.println("Top " + maxPositives + " scores:");
            for (Double d : tmp_scores) System.out.println(d);
            System.out.println("------");

            for (Double d : tmp_scores)
            {
                for (Integer i : positives)
                {
                    if (scores[i] == d) topPositives.add(i);
                }
            }
            System.out.println("Top " + maxPositives + " positions:");
            for (Integer i : topPositives) System.out.println(i);
            System.out.println("------");

            return topPositives;
        }
        else
        {
            return positives;
        }
    }




}