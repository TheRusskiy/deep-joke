'use strict';

angular.module("psycho").factory('ShowError', ['blockUI', 'SweetAlert', function ShowError(blockUI, SweetAlert) {
  return function(response) {
    console.log(response.data);
    blockUI.stop();
    SweetAlert("Error", response.data.message || 'Some Error Occurred', "error")
  };
}]);
