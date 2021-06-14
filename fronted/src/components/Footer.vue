<template>
  <div id="footer">
    <div id="footer-content">
      <p id="footer-text">{{msg}}</p>
    </div>
  </div>
</template>

<script>
export default {
    name: "Footer",
    data() {
        return {
            msg: "@Copyright XMU ***"
        };
    }
}
</script>

<style scoped>
  #footer {
    overflow: hidden;
    background-color: #333333;
    text-align: center;
  }
  #footer-text {
    color: #aaaaaa;
  }
</style>
