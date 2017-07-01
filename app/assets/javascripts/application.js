// This is a manifest file that'll be compiled into application.js, which will include all the files
// listed below.
//
// Any JavaScript/Coffee file within this directory, lib/assets/javascripts, vendor/assets/javascripts,
// or any plugin's vendor/assets/javascripts directory can be referenced here using a relative path.
//
// It's not advisable to add code directly here, but if you do, it'll appear at the bottom of the
// compiled file. JavaScript code in this file should be added after the last require_* statement.
//
// Read Sprockets README (https://github.com/rails/sprockets#sprockets-directives) for details
// about supported directives.
//
//= require jquery
//= require jquery_ujs
//= require jquery-ui.min
//= require bootstrap
//= require jquery.mCustomScrollbar
//= require input-mask.min
//= require bootstrap-tokenfield.min
//= require superflat
//= require jquery.remotipart
//= require jquery.flot
//= require jquery.flot.resize
//= require jquery.flot.orderBars
//= require curvedLines
//= require jquery.sparkline.min
//= require jquery.easypiechart.min
//= require charts
//= require curved-line-chart
//= require line-chart
//= require pie-chart
//= require 'lodash/lodash.js'
//= require 'angular/angular.js'
//= require 'angular-block-ui/dist/angular-block-ui.js'
//= require 'ui-select/dist/select.js'
//= require 'angular-sanitize/angular-sanitize.js'
//= require 'angular-ui-bootstrap/dist/ui-bootstrap-tpls.js'
//= require 'sweetalert2/dist/sweetalert2.js'
//= require 'checklist-model/checklist-model.js'
//= require_tree ./application

$(document).ready(function() {
  $('*[data-mask]').each(function() {
      var input = $(this),
          options = {};

      if (input.attr('data-mask-reverse') === 'true') {
          options.reverse = true;
      }

      if (input.attr('data-mask-maxlength') === 'false') {
          options.maxlength = false;
      }

      input.mask(input.attr('data-mask'), options);
  });
});