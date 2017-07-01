app = angular.module('psycho', [
  'blockUI'
  'ui.select'
  'ngSanitize'
  'ui.bootstrap'
  'checklist-model'
]);

app.config ($httpProvider, uiSelectConfig) ->
  authToken = angular.element("meta[name=\"csrf-token\"]").attr("content")
  $httpProvider.defaults.headers.common["X-CSRF-TOKEN"] = authToken
  uiSelectConfig.theme = 'bootstrap';
