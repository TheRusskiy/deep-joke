angular.module('psycho').controller 'JokeCtrl', ($scope, $http, blockUI, ShowError, SweetAlert)->

  $scope.getJoke = (text)->
    blockUI.start()
    $http.post('/joke', {
      text: text
    }).then ({ data: { joke: joke } })->
      blockUI.stop()
      rnd = Math.round((Math.random() * 8))
      pic = "pics/#{rnd}.jpeg"
      html = "<div><em class='joke-text'>\"#{joke}\"</em><br/><img src='#{pic}'></div>"
      SweetAlert(
        title: "Deep Petrosyan says:",
        html: html
        type: null
        confirmButtonText: 'Haha!'
      )
    , ShowError