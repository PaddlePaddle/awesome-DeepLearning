<!DOCTYPE html>
<html lang="zh">
<head>
    <title>百度认证平台</title>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"/>
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black">
    <meta name="viewport" content="width=device-width, maximum-scale=1.0, user-scalable=0, initial-scale=1.0">
    <meta content="telephone=no" name="format-detection">

    <link rel="icon" href="/ico/favicon.ico" type="image/x-icon"/>
    <link rel="shortcut icon" href="/ico/favicon.ico" type="image/x-icon"/>
    <link rel="bookmark" href="/ico/favicon.ico" type="image/x-icon"/>
    <link rel="stylesheet" href="/css/base.css?v=2.0">
    <link rel="stylesheet" href="/css/login.css?v=2.0">
    <link rel="stylesheet" href="/css/action.min.css?v=2.0">
</head>

<body>

<div class="wrap">

    <div class="header">
        <a href="/login">
            <img class="logo" src="/images/logo.png">
        </a>

        <div class="more">
            <img src="/images/more.svg">
        </div>

        <div class="right">
            <a href="/login?service=https%3A%2F%2F82011.icoding.baidu-int.com%2Fplatform%2Fuser%2Fstokendecrypt%3Fcallback%3Dhttps%3A%2F%2F82011.icoding.baidu-int.com%2F&amp;appKey=uuapclient-17-Q2TW0PsuYElPAhSDGwIT&amp;locale=en">EN</a>
            
            
        </div>
    </div>

    <div class="h5-nav hide">
        <div class="container">
            
            <a href="https://eac.baidu-int.com/#/pwd/reset">忘记密码</a>
            <a href="/login?service=https%3A%2F%2F82011.icoding.baidu-int.com%2Fplatform%2Fuser%2Fstokendecrypt%3Fcallback%3Dhttps%3A%2F%2F82011.icoding.baidu-int.com%2F&amp;appKey=uuapclient-17-Q2TW0PsuYElPAhSDGwIT&amp;locale=en">EN</a>
            
            
            <a href="/manage/help">帮助</a>
            <s>
                <i></i>
            </s>
        </div>
    </div>

    <div class="shade">
        <img src="/images/loginSuccess/wait.gif"/>
        <p>正在登录</p>
    </div>

    <div class="login">
        <div class="content">
            <div class="box">
                <div class="loading"></div>
                <div class="toast-wrap">
                    <span class="toast-msg">网络超时,请刷新重试</span>
                </div>
                <div class="tooltip">
                    <div class="tooltip-arrow"></div>
                    <div class="tooltip-inner">
                        <div>请保证手机如流版本</div>
                        <div>IOS版本在1.0.0以上</div>
                        <div>Android在1.9.9以上</div>
                    </div>
                </div>
                <div class="nav">
                    <div class="h5-title">账号密码登录</div>
                    <span class="tab on" data-type="email" id="1">账号密码登录</span>
                    <span class="line">|</span>
                    <span class="tab" data-type="scan" id="2">扫码登录</span>
                    <span class="line">|</span>
                    <span class="tab" data-type="token" id="3">Token登录</span>
                </div>

                <form method="post" id="form-email" action="/login">
                    <div class="email-area">
                        
                        
                        <div class="li list text username">
                            <input type="text" id="username" data-type="username" name="username" maxlength="90"
                                   value=""
                                   placeholder="百度员工账号"/>
                        </div>
                        <div class="li list text password">
                            <input type="password" id="password-email" data-type="password"
                                   placeholder='账号密码'>
                        </div>
                        <div class="li attach">
                            <span class="checkbox check"></span>
                            <span>自动登录</span>
                        </div>

                        <div class="li hint">
                            <em>
                                
                            </em>
                        </div>

                        <div class="li bt-login commit" id="emailLogin">
                            <span>登录</span>
                        </div>

                        <div class="li changeLoginType">
                            <span class="show-actions">切换登录方式</span>
                        </div>

                        <div class="li other">
                        <span class="help">
                            
                            <a target="_blank" href="https://eac.baidu-int.com/#/pwd/reset">忘记密码</a>
                            <a href="/manage/help" target="_blank">帮助</a>
                        </span>
                        </div>
                        <input type="hidden" name="password" id="encrypted_password_email" value=''/>
                        <input type="hidden" name="rememberMe" value="on">
                        <input type="hidden" name="lt" id="lt-email" value="LT-631912788680781825-q9xsj">
                        
                        <input type="hidden" name="execution" value="e6s1">
                        <input type="hidden" name="_eventId" value="submit">
                        <input type="hidden" value='1' name="type">
                    </div>
                </form>

                <form method="post" id="form-token" action="/login">
                    <div class="token-area">
                        
                        
                        <div class="li list text username">
                            <input type="text" id="token" data-type="username" name="username" maxlength="90"
                                   value=""
                                   placeholder="百度员工账号">
                        </div>
                        <div class="li list text password">
                            <input type="password" id="password-token" data-type="password"
                                   placeholder="PIN+RSA(RSA Token)动态码">
                        </div>
                        <div class="li attach" style="display: none">
                            <span class="checkbox"></span>
                            <span>自动登录</span>
                        </div>

                        <div class="li hint">
                            <em>
                                
                            </em>
                        </div>

                        <div class="li bt-login commit" id="tokenLogin">
                            <span>登录</span>
                        </div>

                        <div class="li changeLoginType">
                            <span class="show-actions">切换登录方式</span>
                        </div>

                        <div class="li other">
                        <span class="help">
                            <a href="/manage/help" target="_blank">帮助</a>
                        </span>
                        </div>
                        <input type="hidden" name="password" id="encrypted_password_token" value=''/>
                        <input type="hidden" name="rememberMe" value="on">
                        <input type="hidden" name="lt" id="lt-token" value="LT-631912788680781825-q9xsj">
                        
                        <input type="hidden" name="execution" value="e6s1">
                        <input type="hidden" name="_eventId" value="submit">
                        <input type="hidden" value='3' name="type">
                    </div>
                </form>

                <form method="post" id="formQRCode" action="/login">
                    <div class="qcode-area">
                        <div class="qcode" id="qcode">
                        </div>
                        <div class="scan-success">
                        </div>
                        <div class="li hint">
                            <em>
                                
                            </em>
                        </div>
                        <div class="li changeLoginType">
                            <span class="show-actions">切换登录方式</span>
                        </div>
                        <input type="hidden" name="username" maxlength="90" id="qrCodeUsername">
                        <input type="hidden" name="password" id="qrCodePassword">
                        <input type="hidden" name="rememberMe" value="on">
                        <input type="hidden" name="lt" id="lt-qrCode" value="LT-631912788680781825-q9xsj">
                        
                        <input type="hidden" name="execution" value="e6s1">
                        <input type="hidden" name="_eventId" value="submit">
                        <input type="hidden" value='2' name="type">
                    </div>
                </form>
            </div>
        </div>
    </div>
    
</div>

<script src="/js/lib/flex.min.js?v=2.0"></script>
<script type="text/javascript" src="/js/lib/jquery3.2.1.min.js"></script>
<script type="text/javascript" src="/js/lib/jquery.placeholder.min.js"></script>
<script type="text/javascript" src="/js/jsencrypt.min.js"></script>
<script type="text/javascript" src="/js/lib/actions.min.js?v=2.0"></script>
<script type="text/javascript" src="/js/login.js?v=6.0"></script>
<script type="text/javascript" src="/js/header.js?v=2.0"></script>
<script type="text/javascript"
        src="/beep-sdk.js?language=zh&amp;v=1628589349194"></script>


<script type="text/javascript">
    var notnull = "\u8F93\u5165\u4E0D\u80FD\u4E3A\u7A7A!",
        sp_noemail = "\u8D26\u53F7\u4E0D\u5305\u62EC\u90AE\u7BB1\u540E\u7F00\uFF0C\u5982@baidu.com",
        sp_username = "\u767E\u5EA6\u5458\u5DE5\u8D26\u53F7",
        sp_passwd = "\u8D26\u53F7\u5BC6\u7801",
        sp_hardToken = "PIN+RSA(RSA Token)\u52A8\u6001\u7801",
        usernameformaterror = "\u8D26\u53F7\u683C\u5F0F\u9519\u8BEF!",
        usernameprompt = "\u767E\u5EA6\u5458\u5DE5\u8D26\u53F7",
        lastLoginType = 1,
        securityLevel = 2,
        rsaPublicKey = "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDSzTSkeLSG1wAOAMRh4L4O78jP4KgSwvMWSnpiWUrOpGknhHMMeoESI94NXdp9DZkptocfuo6dygUOsM+YM60+EVpRg2e9yWApvj88n88+yqQSJeCTRMRS2CDKZrOqf3WOQx7X72Ogj+yTx7mE+Ld+hhrl1ghPxCulQyOnMDSzbwIDAQAB",
        beepQrCodeToken = "6eef72577bf7820f72f71e6ac90d0461f1450bf99014af3c2cacaef55b461410",
        mailBoxLoginTabName = "\u8D26\u53F7\u5BC6\u7801\u767B\u5F55",
        qrCodeLoginTabName = "\u626B\u7801\u767B\u5F55",
        mobileHiLoginTabName = "\u624B\u673A\u5982\u6D41\u767B\u5F55",
        hardTokenLoginTabName = "Token\u767B\u5F55",
        cancelButtonName = "\u53D6\u6D88";
</script>

</body>
</html>
