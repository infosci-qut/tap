/*
 * Copyright 2016-2017 original author or authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controllers

import java.io.File
import java.nio.file.{Files, Path, Paths}
import java.util.{Base64, UUID}

import akka.stream.IOResult
import akka.stream.scaladsl.{FileIO, Sink}
import akka.util.ByteString
import play.api.Logger
import play.api.libs.streams._
import play.api.mvc.InjectedController
import play.api.mvc.MultipartFormData.FilePart
import play.core.parsers.Multipart.FileInfo

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

/**
  * Created by andrew@andrewresearch.net on 22/8/17.
  */

/**
  * Handles all requests for uploading and managing file.
  * Currently supports extensions txt, jar, zip and no extension.
  */

class FileController extends InjectedController {

  // Sepearates the suffix, random number and prefix, and is
  // used to facilitate later extraction of the random number
  // that is generated by createTempFile
  val generatedFileDelimiter = '|'

  // The upload action is based on "https://github.com/playframework/play-scala-fileupload-example"
  // and the "Handling File upload - Writing your own body parser" in the
  // the play framework documentation:
  // "https://www.playframework.com/documentation/2.6.x/ScalaFileUpload"
  // The custom loader is considerably more complex to understand, and I
  // cannot find a documentation source where it is explained fully.
  // I am using it to allow for more flexibility in extending this feature.
  //
  def upload = Action(parse.multipartFormData(handleFilePartAsFile)) { implicit request =>

    Logger.trace("Got upload request from:" + request.remoteAddress)
    val fileId = request.body.file("name").map {

      case FilePart(key, filename, contentType, file, fileSize, dispositionType)  =>
        Logger.info("key=" + key + ",filename=" + filename + ",contentType=" + contentType + ",file= " + file)
        val data = getFileId(file, filename)
        data
    }
    Logger.info("file id ="+fileId)
    Ok(s"file id = ${fileId}")
  }

  type FilePartHandler[A] = FileInfo => Accumulator[ByteString, FilePart[A]]


  //
  // Custom upload file handler
  // uploads the file and copies its contents into a file that has
  // the path, prefix and suffix used below.
  //
  // The path, prefix are known. The suffix is limited to a range
  // of file suffixes that we support, so only the generated
  // unique id is variable. Since that number is
  // part of the id that is returned to the user, we should be able
  // to locate the file when we are provided with the id,
  // without the need for a mapping mechanism.
  //
  // Currently the uploaded files are stored in /tmp directory
  // which means they will be cleaned up when the server machine is
  // recycled and we don't have to worry about a clean up mechanism.
  // The downside is that the interval of cleanup is variable, and
  // there is a chance that we will run out of server disk space if
  // the server machine is not cycled for a long time.
  //
  private def handleFilePartAsFile: FilePartHandler[File] = {

    case FileInfo(partName, filename, contentType, dispositionType)      =>

      Logger.trace("-->handleFilePartAsFile, uploading file")


      val prefix = "TAP-$upload$-$temp$" + generatedFileDelimiter
      val suffix = getSupportedFileExtension(filename) match {
        case Some(ext) => generatedFileDelimiter + "." + ext
        case None => ""
      }
      val id = UUID.randomUUID().toString.getBytes
      val b64id = Base64.getEncoder.encodeToString(id)
      val upload_dir = "/tmp/"
      val pathname = upload_dir + prefix + b64id + suffix
      new File(pathname).createNewFile()
      val path:Path = Paths.get(pathname)
      Logger.info("path=" + path)
      val fileSink: Sink[ByteString, Future[IOResult]] = FileIO.toPath(path)
      val accumulator: Accumulator[ByteString, IOResult] = Accumulator(fileSink)

      accumulator.map {
        case IOResult(count, status) =>
          Logger.info("count = " + count + ",status=" + status)
          FilePart(partName, filename, contentType, path.toFile)
      }
  }

  //
  // Assumes the server uploaded file has been created.
  // The returned id is the base64 id generated during creating the uploaded file,
  // This id is part of the file name and can be used to relocate the file.
  //
  //
  private def getFileId(file: File, fileName: String) = {
    Logger.trace("-->getFileId")
    val size = Files.size(file.toPath)
    Logger.debug("size = " + size)
    val fileID = getFileUniqueId(file.toPath.toString)
    Logger.trace("<--getFileId")
    fileID
  }

  //
  // Currently, the supported extensions are txt, jar, zip and no extension
  // These are the only extension that we will try when locating
  // an uploaded file.
  //
  def getSupportedFileExtension(fileName:String): Option[String] =
  {
    Logger.trace("-->getSupportedFileExtension fileName=" + fileName)
    val suffix = getFileExtension(fileName)
    suffix match {
      case Some("txt") => Some("txt")
      case Some("zip") => Some("zip")
      case Some("jar") => Some("jar")
      case Some(ext) => None
      case None => None
    }
  }

  //
  // Returns the extension of the file that is being uploaded.
  // We preserve this extension when we are creating the uploaded file
  // for supported extension types.
  // The uploaded file name on the other hand is not used.
  //
  def getFileExtension(fileName:String) : Option[String] = {
    Logger.trace("-->getFileExtension fileName=" + fileName)
    val parts = fileName split('.')
    if (parts.length >= 2) Some(parts(parts.length - 1)) else None
  }

  //
  // This is the base64 generated UUID unique identifier.
  // The delimiters as part of the prefix and suffix helps us
  // locate it easily.
  //
  def getFileUniqueId(fileName:String):String = {
    Logger.trace("-->getFileUniqueId fileName=" + fileName)
    val parts = fileName split(generatedFileDelimiter)
    parts(1)
  }
}




