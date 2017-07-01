'use strict';

angular.module("psycho").factory('SweetAlert', ['$window', function SweetAlert($window) {
    return $window.swal;
  }]);
