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

package controllers

import javax.inject.Inject
import play.api.mvc.{Action, AnyContent, InjectedController}
import views.{HomePage, QueriesPage}

/**
  * Created by andrew@andrewresearch.net on 22/8/17.
  */

/**
  * Handles all requests to root site. [views.HomePage$] and [views.QueriesPage$]
  *
  * @param assets used to find assets according to configured base path and URL base.
  */

class DefaultController @Inject() (assets: AssetsFinder) extends InjectedController {


  /**
    * Provide or GET HomePage to client
    *
    * @return [[play.api.mvc.Action Action]] of type [[AnyContent]]
    */
  def index:Action[AnyContent] = Action {
    //Ok(views.html.index())
    Ok(HomePage.render("TAP - Text Analytics Pipeline"))
  }

  /**
    * Provide or GET QueriesPage to client
    *
    * @return [[play.api.mvc.Action Action]] of type [[AnyContent]]
    */
  def queries:Action[AnyContent] = Action {
    Ok(QueriesPage.render("TAP - Example Queries"))
  }
}
