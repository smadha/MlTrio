package edu.usc.mltrio.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * Find best num_neigh and similarity matrix using validation
 * 
 * LOWER THE BETTER
 * 
 * numFeatures - 64 lambda - 1.0E-4 Acc - 0.3511877294688672
 * numFeatures - 64 lambda - 0.001 Acc - 0.19067660161192174
 * 
 * numFeatures - 16 lambda - 1.0E-5 Acc - 1.9206223071447528
numFeatures - 16 lambda - 1.0E-4 Acc - 0.4778584128588928
numFeatures - 16 lambda - 0.001 Acc - 0.23063901276558188
numFeatures - 16 lambda - 0.01 Acc - 0.15821677404183898
numFeatures - 32 lambda - 1.0E-5 Acc - 1.3967344276030973
numFeatures - 32 lambda - 1.0E-4 Acc - 0.3938753453566021
numFeatures - 32 lambda - 0.001 Acc - 0.22004050429783453
numFeatures - 32 lambda - 0.01 Acc - 0.1508279857389991
numFeatures - 64 lambda - 1.0E-5 Acc - 1.860971584323923
numFeatures - 64 lambda - 1.0E-4 Acc - 0.40138566167735484
numFeatures - 64 lambda - 0.001 Acc - 0.19399877313013533
numFeatures - 64 lambda - 0.01 Acc - 0.14611190250341966
numFeatures - 128 lambda - 1.0E-5 Acc - 1.1303028809706241
numFeatures - 128 lambda - 1.0E-4 Acc - 0.30393174377740173
numFeatures - 128 lambda - 0.001 Acc - 0.1826482290884484
numFeatures - 128 lambda - 0.01 Acc - 0.15133927458241486
 */
public class MahoutMatFactRecommender implements RecommenderBuilder{
	
	int numFeatures = 50; 
	double lambda = 0.1; 
	int numIterations = 50;
	
	public MahoutMatFactRecommender(int numFeatures, double lambda, int numIterations) {
		this.numFeatures = numFeatures;
		this.lambda = lambda;
		this.numIterations = numIterations;
	}

	public static void main(String[] args) throws Exception{
		DataModel model = new FileDataModel(new File(Constants.INVITED_INFO_TRAIN_MAHOUT_CSV));
		
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		
		int [] numFeaturesArr = {50,64};
		double [] lambdaArr = {0.1, 1, 1.5};
		int[] iterArr = {20,70,100};
				
		for(int numFeaturesI : numFeaturesArr){
			for(double lambdaI : lambdaArr ){
				for (int iterI : iterArr ){
					RecommenderBuilder builder = new MahoutMatFactRecommender(numFeaturesI, lambdaI, iterI);
					
					double result = evaluator.evaluate(builder, null, model, 0.8, 0.8);
					System.out.println("numFeatures - "+ numFeaturesI + " lambda - "+ lambdaI + " iter - "+ iterI + " Acc - " + result);
				}
			}
		}
	}

	public Recommender buildRecommender(DataModel model) throws TasteException {

		ALSWRFactorizer factorizer = new ALSWRFactorizer(model, numFeatures, lambda, numIterations);

		return new SVDRecommender(model, factorizer);
	}
}
