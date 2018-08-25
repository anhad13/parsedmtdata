import edu.stanford.nlp.util.logging.Redwood;

import java.io.StringReader;
import java.util.List;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.Tree;
import java.io.File;
import java.util.Scanner;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import java.io.FileNotFoundException;
import edu.stanford.nlp.sentiment.CollapseUnaryTransformer;




import edu.stanford.nlp.parser.*;
import edu.stanford.nlp.parser.lexparser.*;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.stats.ClassicCounter;

import java.util.*;
import java.io.Reader;

import edu.stanford.nlp.sentiment.CollapseUnaryTransformer;
/**
 * Demonstrates how to first use the tagger, then use the
 * ShiftReduceParser.  Note that ShiftReduceParser will not work
 * on untagged text.
 *
 * @author John Bauer
 */
public class ShiftConvertAll  {

  
  /** A logger for this class */
  private static Redwood.RedwoodChannels log = Redwood.channels(ShiftConvertAll.class);
  public static void main(String[] args) throws FileNotFoundException{
          boolean insideFactor = false;
          HeadFinder hf = new ModCollinsHeadFinder();
    TreebankLanguagePack tlp = new PennTreebankLanguagePack();
    boolean mf = false;
    int mo = 1;
    boolean uwl = false;
    boolean uat = false;
    double sst = 20.0;
    boolean mfs = false;
    boolean simpleLabels = false;
    boolean noRebinarization = false;
    String modelPath = "/Users/anhadmohananey/Downloads/edu/stanford/nlp/models/srparser/englishSR.ser.gz";
    String taggerPath = "/Users/anhadmohananey/Downloads/stanford-postagger-2018-02-27/models/english-left3words-distsim.tagger";
    File file = new File("/scratch/am8676/parsedmtdata/en-de/OpenSubtitles2016.de-en.en");

    Scanner sc = new Scanner(file);

    for (int argIndex = 0; argIndex < args.length; ) {
      switch (args[argIndex]) {
        case "-tagger":
          taggerPath = args[argIndex + 1];
          argIndex += 2;
          break;
        case "-model":
          modelPath = args[argIndex + 1];
          argIndex += 2;
          break;
        default:
          throw new RuntimeException("Unknown argument " + args[argIndex]);
      }
    }

    String text = "My dog likes to shake his stuffed chickadee toy.";

    MaxentTagger tagger = new MaxentTagger(taggerPath);
    ShiftReduceParser model = ShiftReduceParser.loadModel(modelPath);
     TreeTransformer tt = new TreeBinarizer(hf, tlp, insideFactor, mf, mo,
             uwl, uat, sst, mfs, simpleLabels, noRebinarization);
    CollapseUnaryTransformer transformer = new CollapseUnaryTransformer();
    while(sc.hasNextLine())
    {
      DocumentPreprocessor tokenizer = new DocumentPreprocessor(new StringReader(sc.nextLine()));
      String sents="";
      for (List<HasWord> sentence : tokenizer) {
        List<TaggedWord> tagged = tagger.tagSentence(sentence);
        Tree tree = model.apply(tagged);
        //log.info(tree);
        Tree newT = transformer.transformTree(tt.transformTree(tree));
        String xs=unlabeledPrint(newT);
        sents=xs;
        //System.out.printf(tree.flatten().pennString());
      }
      System.out.printf(sents);
      System.out.printf("\n");
    }


  }

     static String unlabeledPrint(Tree tree) {
        if (tree.isLeaf()) {
            return tree.nodeString();
        } else if (tree.isPreTerminal()) {
            for (Tree child : tree.children()) {
                return unlabeledPrint(child);
            }
            return "---";
        } else {
            String rv = "(";
            for (Tree child : tree.children()) {
                rv = rv + " " + unlabeledPrint(child); 
            }
            return rv + " )";
        }
    }
} 
