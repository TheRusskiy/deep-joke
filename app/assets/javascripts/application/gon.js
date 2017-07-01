'use strict';

angular.module("psycho").factory('gon', ['$window', function gon($window) {
    return $window.gon;
  }]);
