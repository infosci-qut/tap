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

package io.heta.tap.analysis.batch

import java.util.UUID

import akka.actor.{Actor, ActorRef}
import akka.pattern.ask
import akka.stream.alpakka.s3.MultipartUploadResult
import akka.stream.scaladsl.{Flow, Sink, Source}
import akka.util.{ByteString, Timeout}
import akka.{Done, NotUsed}
import com.typesafe.scalalogging.Logger
import io.heta.tap.analysis.batch.BatchActor.{AnalyseSource, CheckProgress, INIT, ResultMessage}
import io.heta.tap.analysis.clu.CluAnnotatorActor.AnnotateRequest
import io.heta.tap.pipelines.Pipe
import io.heta.tap.pipelines.materialize.FilePipeline.{File, FileInfo}
import io.heta.tap.pipelines.materialize.PipelineContext.executor
import io.heta.tap.pipelines.materialize.{ByteStringPipeline, FilePipeline}
import org.clulab.processors.Document

import scala.concurrent.Future
import scala.concurrent.duration._

/**
  * This actor controls the reading of files from S3, the selection of a pipeline to run, and the writing of the
  * resultant analytics back to S3.
  * It is designed to include basic monitoring to facilitate queries on progress of batch jobs.
  */
object BatchActor {
  object INIT
  sealed trait AnalysisRequest
  case class AnalyseSource(bucket:String,analysisType:String,annotator:ActorRef) extends AnalysisRequest
  case class CheckProgress(bucket:String,batchId:String) extends AnalysisRequest
  case class ResultMessage(result:String,message:String)
}

class BatchActor extends Actor {

  val logger: Logger = Logger(this.getClass)

  val awsS3 = new AwsS3Client()
  //val da = new DocumentAnnotating()

  val parallelism = 5

  /**
    * Receive messages that the actor can handle: Such as INIT, as, and cp
    *
    * @return A [[scala.PartialFunction PartialFunction]] of type [[scala.Any Any]] and Unit[[scala.Unit Unit]]
    */
  def receive: PartialFunction[Any,Unit] = {
    case INIT => sender ! init
    case as: AnalyseSource => sender ! analyse(as.bucket,as.analysisType,as.annotator)
    case cp: CheckProgress => sender ! progress(cp.bucket,cp.batchId)
    case msg:Any => logger.error(s"BatchActor received unkown msg: ${msg.toString}") // scalastyle:ignore
  }

  /**
    * Initialise to connect to S3
    *
    * @return true to allow process to continue
    */
  def init: Boolean = {
    //TODO Check that can connect to S3
    true //for now just return true to allow process to continue
  }

  /**
    * Use specified pipelines to analyse a batch of files from a given bucket name
    *
    * @param bucket the s3 bucket name
    * @param analysisType any valid pipeline query name
    * @param annotator Annotating actor
    * @return A [[scala.concurrent.Future Future]] of type [[ResultMessage]]
    */
  def analyse(bucket:String,analysisType:String,annotator:ActorRef): Future[ResultMessage] = {
    logger.info(s"Started batch $analysisType for bucket: $bucket")
    val batchId = UUID.randomUUID().toString
    createBatchFolder(bucket,batchId)
    val result = getPipeline(analysisType,annotator)
    result match {
      case Right(pipeline) => {
        processFiles(bucket,batchId, pipeline)
      }
      case Left(error) => logger.error(error.getMessage)
    }
    if(result.isRight) {
      Future.successful(ResultMessage(batchId,s"Started $analysisType analysis for bucket $bucket"))
    } else {
      Future.successful(ResultMessage("",result.left.get.getMessage))
    }
  }

  /**
    * Checking for batch progress for bucket
    *
    * @param bucket the s3 bucket name
    * @param batchId the s3 bucket Id
    * @return A [[scala.concurrent.Future Future]] of type [[ResultMessage]]
    */
  def progress(bucket:String,batchId:String): Future[ResultMessage] = {
    logger.info(s"Checking batch progress for: $bucket/$batchId")
    Future.successful(ResultMessage("",s"Progress checked for $bucket/$batchId"))
  }

  /**
    * Create Batch Folder for batch progress for bucket
    *
    * @param bucket the s3 bucket name
    * @param folderName Name of the Batch Folder
    * @return A [[scala.concurrent.Future Future]]
    */
  private def createBatchFolder(bucket:String,folderName:String) = Future {
    val key = s"$folderName/__metadata"
    logger.debug(s"Creating destination: $key")
    ByteStringPipeline(Source.empty[ByteString],awsS3.sinkfileToBucket(bucket,key)).run
  }


  /**
    * Get the desire pipeline type use for analysis
    *
    * @param analysisType any valid pipeline query name
    * @param annotator Annotating actor
    * @returns the desire analysis type, else return Exception message
    */
  private def getPipeline(analysisType:String,annotator:ActorRef):Either[Throwable,Flow[File, File, NotUsed]] = {
    val fields = Pipe.getClass.getDeclaredFields.map(_.getName).toVector
    if (fields.contains(analysisType)) {
      val field = Pipe.getClass.getDeclaredField(analysisType)
      field.setAccessible(true)
      try {
        val segment = field.get(Pipe).asInstanceOf[Flow[Document,File,NotUsed]]
        Right(annotatedDocFromFile(annotator) via segment)
      } catch {
        case error:Exception => Left(error)
      }
    } else {
      Left(new Exception(s"No such analysisType: $analysisType"))
    }
  }

  /**
    * Annotated document from file
    *
    * @param annotator Annotating actor
    * @return A `Flow` is a set of stream processing steps that has one open input and one open output.
    */
  private def annotatedDocFromFile(annotator:ActorRef): Flow[File, Document, NotUsed] = Flow[File]
    .mapAsync[org.clulab.processors.Document](parallelism) { file =>
    logger.debug(s"|${annotator.path.toString}|")
    implicit val timeout: Timeout = 60.seconds
    (annotator ? AnnotateRequest(file.contents.utf8String)).mapTo[Document]
      .map{ doc =>
        doc.id = Some(file.name)
        logger.debug("DOC text:"+doc.text)
        doc.sentences.foreach(s => logger.debug("SENTENCE: "+s.tags.getOrElse(Array()).mkString("|")))
        doc
      }
  }

  /**
    * Processing files for batchId in bucket
    *
    * @param bucket the s3 bucket
    * @param batchId the s3 bucket Id
    * @param pipeline TAP analysis pipeline
    * @return Run FilePipeline
    */
  private def processFiles(bucket:String,batchId:String,pipeline: Flow[File, File, NotUsed]): Future[Future[Done]] = Future {
    val inputFolder = "source_files"
    logger.info(s"Processing files for $batchId in $bucket/$inputFolder")
    val fileSourcer = (fi:FileInfo) => sourceFileFromS3(bucket,fi.name)
    val fileSource = sourceFileInfoFromS3(bucket,inputFolder).flatMapMerge(parallelism, fileSourcer)
    val fileSink = sinkFileToS3(bucket,batchId)
    FilePipeline(fileSource,pipeline,fileSink).run
  }

  /**
    * Source file information from s3
    *
    * @param bucket the s3 bucket
    * @param prefix Prefix of the keys you want to list under passed bucket
    * @return A [[akka.stream.scaladsl.Source Source]] of [[FileInfo]]
    */
  private def sourceFileInfoFromS3(bucket:String,prefix:String): Source[FileInfo, NotUsed] = awsS3.getContentsForBucket(bucket,Some(prefix))
    .filterNot(_.key.endsWith("/"))
    .map(c => FileInfo(c.key,c.size,c.lastModified))

  /**
    * Source file from s3
    *
    * @param bucket the s3 bucket
    * @param key string
    * @return A [[akka.stream.scaladsl.Source Source]] of [[File]]
    */
  private def sourceFileFromS3(bucket:String,key:String): Source[File, NotUsed] = {
    logger.info(s"Reading $bucket/$key")
    awsS3.sourceFileFromBucket(bucket,key)
      .mapAsync[File](4)(bs => bs.map( b => File(key.split("/").last,b)))
  }


  /**
    * Sink file to s3
    *
    * @param bucket the s3 bucket name
    * @param batchId the s3 bucket Id
    * @return A [[akka.stream.scaladsl.Sink Sink]] of [[File]] and [[Future[Done]]]
    */
  private def sinkFileToS3(bucket:String,batchId:String): Sink[File, Future[Done]] = Sink.foreachAsync[File](parallelism){ f =>
    val key = newFileName(s"$batchId/${f.name}","result")
    logger.info(s"Writing: $bucket/$key")
    val sink: Sink[ByteString, Future[MultipartUploadResult]] = awsS3.sinkfileToBucket(bucket,key)
    Future(ByteStringPipeline(Source.single[ByteString](f.contents),sink).run)
  }

  /**
    * Create a new file name
    *
    * @param current current
    * @param tag tag
    * @param inExt input extension
    * @param outExt output extension
    * @return Create new file name
    */
  private def newFileName(current:String,tag:String,inExt:String=".txt",outExt:String=".json"): String =
    current.dropRight(inExt.length).concat(s"-$tag$outExt")

}
