/*
 * Copyright (c) 2016-2018 original author or authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
 * language governing permissions and limitations under the License.
 *
 */

package controllers.handlers

import akka.actor.ActorRef
import akka.pattern.ask
import akka.util.Timeout
import io.heta.tap.analysis.clu.CluAnnotatorActor.INIT
import io.heta.tap.data.doc.expression.affect.AffectThresholds
import io.heta.tap.data.results._
import io.heta.tap.pipelines.AnnotatingTypes._
import io.heta.tap.pipelines.materialize.TextPipeline
import io.heta.tap.pipelines.{Annotating, Cleaning}
import javax.inject.{Inject, Named}
import models.graphql.Fields._
import play.api.Logger

import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.Success

/**
  * Created by andrew@andrewresearch.net on 20/2/17.
  */

/**
  * TextAnalysisHandler handles all text analysis requests
  *
  * @param clean a query that will clean and format the text depending on which parameters you pass
  * @param cluAnnotator support sending messages to the actor it represents
  */

class TextAnalysisHandler @Inject() (clean: Cleaning, @Named("cluAnnotator")cluAnnotator:ActorRef) extends GenericHandler {

  import io.heta.tap.pipelines.materialize.PipelineContext.executor

  val logger: Logger = Logger(this.getClass)

  val annotate = new Annotating(cluAnnotator)

  private val pipe = annotate.Pipeline

  {
    implicit val timeout: Timeout = 60.seconds
    (cluAnnotator ? INIT).onComplete{
      case Success(result:Boolean) => if(result) {
        logger.info("CluAnnotatorActor initialised successfully")
      } else {
        logger.error("There was a problem initialising the CluAnnotatorActor")
      }
      case scala.util.Failure(exception) => logger.error(exception.getMessage)
    }
  }

  /**
    * Clean is a query that will clean and format the text depending on which parameters you pass
    *
    * @param text Optional version texts
    * @param parameters Optional version parameters
    * @param start
    * @return A [[scala.concurrent.Future Future]] of with type [[StringResult]]
    */
  def clean(text: Option[String], parameters: Option[String], start:Long): Future[StringResult] = {
    Logger.warn(s"TEXT: $text")
    val inputStr = text.getOrElse("")
    val cleanType = extractParameter[String]("cleanType",parameters)
    Logger.info(s"cleanType: $cleanType - text: ${inputStr.take(10)}")
    val outputStr:Future[String] = cleanType match {
      case Some("visible") => TextPipeline(inputStr,clean.Pipeline.revealInvisible).run
      case Some("minimal") => TextPipeline(inputStr,clean.Pipeline.utfMinimal).run
      case Some("simple") => TextPipeline(inputStr,clean.Pipeline.utfSimplify).run
      case Some("preserve") => TextPipeline(inputStr,clean.Pipeline.lengthPreserve).run
      case Some("ascii") =>  TextPipeline(inputStr,clean.Pipeline.asciiOnly).run
      case _ => Future("")
    }
    outputStr.map(StringResult(_,querytime = queryTime(start)))
  }

  /**
    * Annotation is a query that will splitup the text into json data,
    * including seperating the sentences into their own array and providing various stats on each word.
    *
    * @param text Optional version texts
    * @param parameters Optional version parameters
    * @param start
    * @return A [[scala.concurrent.Future Future]] of with type [[SentencesResult]]
    */
  def annotations(text:Option[String],parameters:Option[String],start:Long):Future[SentencesResult] = {
    val inputStr = text.getOrElse("")
    val pipeType = extractParameter[String]("pipeType",parameters)
    Logger.debug(s"STRING: $pipeType")
    val analysis = pipeType match {
      case Some(CLU) => TextPipeline(inputStr, annotate.build(CLU,pipe.cluSentences)).run
      case Some(STANDARD) => TextPipeline(inputStr, annotate.build(STANDARD,pipe.sentences)).run
      case Some(FAST) => TextPipeline(inputStr, annotate.build(FAST,pipe.sentences)).run
      case Some(NER) => TextPipeline(inputStr, annotate.build(NER,pipe.sentences)).run
      case _ => TextPipeline(inputStr, annotate.build(FAST,pipe.sentences)).run
    }
    analysis.map(SentencesResult(_,querytime = queryTime(start)))
  }

  /** A query that will extract the epistemic expressions of a sentence @See [[models.graphql.FieldDocs]*/
  def expressions(text:Option[String],parameters:Option[String],start:Long):Future[ExpressionsResult] =
    TextPipeline(text.getOrElse(""),annotate.build(DEFAULT,pipe.expressions)).run
      .map(ExpressionsResult(_,querytime = queryTime(start)))
  /** A query that will return the syllable count for each word in a sentence and group each sentence into its own array @See [[models.graphql.FieldDocs]*/
  def syllables(text:Option[String],parameters:Option[String],start:Long):Future[SyllablesResult] =
    TextPipeline(text.getOrElse(""),annotate.build(DEFAULT,pipe.syllables)).run
      .map(SyllablesResult(_,querytime = queryTime(start)))
  /** A query that will return the spelling mistakes and possible suggestions for what the intended word was @See [[models.graphql.FieldDocs] */
  def spelling(text:Option[String],parameters:Option[String],start:Long):Future[SpellingResult] =
    TextPipeline(text.getOrElse(""),annotate.build(DEFAULT,pipe.spelling)).run
      .map(SpellingResult(_,querytime = queryTime(start)))
  /** A query that returns the stats on the vocabulary used, it groups them by unique words and how many times they were used
    * @See [[models.graphql.FieldDocs]*/
  def vocabulary(text:Option[String],parameters:Option[String],start:Long):Future[VocabularyResult] =
    TextPipeline(text.getOrElse(""),annotate.build(DEFAULT,pipe.vocab)).run
      .map(VocabularyResult(_,querytime = queryTime(start)))
  /** A query that will return various stats on the text that was parsed @See [[models.graphql.FieldDocs] */
  def metrics(text:Option[String],parameters:Option[String],start:Long):Future[MetricsResult]  =
    TextPipeline(text.getOrElse(""),annotate.build(DEFAULT,pipe.metrics)).run
      .map(MetricsResult(_,querytime = queryTime(start)))
  /** A query that will return the verb, noun and adjective distribution ratios of the sentences @See [[models.graphql.FieldDocs]*/
  def posStats(text:Option[String],parameters:Option[String],start:Long):Future[PosStatsResult] =
    TextPipeline(text.getOrElse(""),annotate.build(DEFAULT,pipe.posStats)).run
      .map(PosStatsResult(_,querytime = queryTime(start)))
  /** a query that will return various stats about the text @See [[models.graphql.FieldDocs] */
  def reflectExpressions(text:Option[String],parameters:Option[String],start:Long):Future[ReflectExpressionsResult] =
    TextPipeline(text.getOrElse(""),annotate.build(DEFAULT,pipe.reflectExpress)).run
    .map(ReflectExpressionsResult(_, querytime = queryTime(start)))

  /** Affect Expressions is a query that will return stats about the valence, arousal and dominance language used @See [[models.graphql.FieldDocs]*/
  def affectExpressions(text:Option[String],parameters:Option[String] = None,start:Long): Future[AffectExpressionsResult] = {
    val thresholds = extractAffectThresholds(parameters)
    Logger.info(s"thresholds: $thresholds")
    TextPipeline(text.getOrElse(""),annotate.build(CLU,pipe.affectExpress(thresholds))).run
      .map(AffectExpressionsResult(_,"",queryTime(start)))
  }


  /**
    * Extracts different affect thresholds to describe and measure emotional states (valence, arousal, dominance)
    *
    * @param parameters Opetional version parameters
    * @return A [[scala.Option Option]] of type [[AffectThresholds]]
    */
  private def extractAffectThresholds(parameters:Option[String]):Option[AffectThresholds] = {
    for {
      v <- extractParameter[Double]("valence",parameters)
      a <- extractParameter[Double]("arousal",parameters)
      d <- extractParameter[Double]("dominance",parameters)
    } yield AffectThresholds(v.asInstanceOf[Double],a.asInstanceOf[Double],d.asInstanceOf[Double])
  }


  //TODO To be implemented
  def shape(text:String):Future[StringResult]   = dummyResult(text)


}
