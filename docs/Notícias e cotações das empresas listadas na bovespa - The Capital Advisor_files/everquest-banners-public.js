jQuery(document).ready(function($) {
    $('.everquest-banner-block').each(function(index, el) {
        var data = {};
        var block = $(this);
        if (block.attr('data-group')) {
            data.group = $(this).attr('data-group');
        }
        if (block.attr('data-banner')) {
            data.banner = $(this).attr('data-banner');
        }

        if (block.attr('data-type')) {
            data.type = $(this).attr('data-type');
        }
        $.get(everquest_banners_rest_endpoint + 'render', data, function(data) {

            block.html(data.html);
            block.attr('data-rendered-banner', data.bid);
            block.attr('data-href', data.link);
        });
        console.log(data);
    });
});